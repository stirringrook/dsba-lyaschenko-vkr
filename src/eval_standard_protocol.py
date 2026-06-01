"""Standard-protocol evaluator for CREMA-D / RAVDESS.

Re-evaluates trained Stage 3 / interim checkpoints under the protocol
that the published audiovisual literature actually uses:
**clip-level accuracy, macro-F1 and weighted-F1 over the native
6 (CREMA-D) / 7 or 8 (RAVDESS) emotion classes**. The per-frame
Aff-Wild2-aligned protocol implemented in :mod:`src.eval_fusion` is
left intact; this module is additive.

Why this exists:
    The interim sweep is reported under a per-frame Aff-Wild2 protocol
    (8-class macro-F1 EXPR + circumplex-CCC VA), which was a deliberate
    choice for in-house consistency but precludes direct comparison
    with AVT-CA, MAG-BERT, MFM, AVERFormer, VAVL, etc. This evaluator
    runs the SAME checkpoints under the published protocol so the two
    numbers can sit side by side.

Aggregation procedure:
    1. Frame-by-frame inference over the validation split (re-uses
       :func:`src.eval_fusion._collect_predictions`).
    2. Per-frame softmax probabilities are averaged into one 8-D vector
       per clip (videoname).
    3. The argmax is restricted to the dataset's native AffWild2-index
       subset (see :mod:`src.datasets.label_mapping`).
    4. Ground-truth labels are recovered from the clip filename, NOT
       from the AffWild2-mapped annotation file --- otherwise RAVDESS
       Calm collapses into Neutral and the supervisor's criticism
       persists.
    5. Accuracy / macro-F1 / weighted-F1 are computed with scikit-learn
       at clip level.

Usage examples::

    # Single trained checkpoint (F1-F5, visual-only, audio-only):
    python -m src.eval_standard_protocol \\
        --config configs/interim/crema_f4_xattn.yaml \\
        --checkpoint results/interim/crema_f4_xattn/best.pt \\
        --dataset cremad

    # F0 post-hoc late blend:
    python -m src.eval_standard_protocol --mode f0_grid \\
        --visual-checkpoint results/interim/crema_visual_only/best.pt \\
        --audio-checkpoint  results/interim/crema_audio_only/best.pt \\
        --config configs/interim/crema_f0_grid.yaml \\
        --dataset cremad

    # Sweep every checkpoint in results/interim/ and write a summary:
    python -m src.eval_standard_protocol --run-all \\
        --interim-results-dir results/interim \\
        --output results/interim/standard_protocol_summary.md
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.datasets.bimodal import BimodalFrameDataset, BimodalWindowDataset
from src.datasets.label_mapping import (
    LABEL_SPACE_CREMAD6,
    LABEL_SPACE_RAVDESS7,
    LABEL_SPACE_RAVDESS8,
    LabelSpace,
    aggregate_softmax_per_clip,
    detect_dataset_from_videoname,
    native_labels_for_clips,
    restricted_argmax,
)
from src.fusion.base import FusionConfig
from src.train_fusion import build_model, wants_window


# ---------------------------------------------------------------------------
# Lightweight prediction collection (re-implemented to also expose videonames)
# ---------------------------------------------------------------------------

def _collect_frame_predictions(
    model, loader, device: str
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Return ``(expr_logits, videonames)`` aligned over the val split.

    Only the EXPR branch is needed for the standard-protocol metric;
    VA / AU outputs are intentionally ignored.
    """
    expr_logits: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            e, _v, _a = model(
                batch["v_feat"].to(device),
                batch["v_scores"].to(device),
                batch["a_feat"].to(device),
            )
            expr_logits.append(e.cpu().numpy())
    logits = np.concatenate(expr_logits, axis=0)
    videonames = tuple(vn for vn, _ in loader.dataset.videoname_frames)
    if len(videonames) != logits.shape[0]:
        raise RuntimeError(
            f"Internal alignment error: {len(videonames)} videonames vs "
            f"{logits.shape[0]} predicted frames."
        )
    return logits, videonames


def _build_val_loader(cfg, variant: str, batch_size: Optional[int] = None) -> DataLoader:
    ds_cls = BimodalWindowDataset if wants_window(variant) else BimodalFrameDataset
    extra = {"window": int(cfg.fusion.get("window", 5))} if wants_window(variant) else {}
    ds = ds_cls(
        annotations_file=cfg.data.val_annotations,
        visual_cache_dir=cfg.visual.features_cache,
        audio_cache_dir=cfg.audio.aligned_dir,
        **extra,
    )
    bs = int(batch_size if batch_size is not None else cfg.train.batch_size)
    return DataLoader(ds, batch_size=bs, shuffle=False)


def _select_label_spaces(dataset: str) -> Tuple[LabelSpace, ...]:
    """Pick the native label space(s) for a dataset.

    For RAVDESS we always evaluate both 7-class (apples-to-apples) and
    8-class (strict, with Calm misses) so the report can be honest about
    the cost of the Calm collapse.
    """
    d = dataset.lower()
    if d in ("cremad", "crema-d", "crema_d"):
        return (LABEL_SPACE_CREMAD6,)
    if d in ("ravdess",):
        return (LABEL_SPACE_RAVDESS7, LABEL_SPACE_RAVDESS8)
    raise ValueError(f"Unsupported dataset '{dataset}'.")


def _autodetect_dataset(videonames: Sequence[str]) -> str:
    """Detect 'cremad' / 'ravdess' from a sample of videonames."""
    sample = next(iter(videonames))
    return detect_dataset_from_videoname(sample)


# ---------------------------------------------------------------------------
# Standard-protocol metric
# ---------------------------------------------------------------------------

@dataclass
class StandardProtocolResult:
    """Standard-protocol scorecard for one (checkpoint, label_space) pair."""

    variant: str
    label_space: str
    n_clips: int
    accuracy: float
    f1_macro: float
    f1_weighted: float
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    per_class_support: Dict[str, int] = field(default_factory=dict)
    confusion_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    notes: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "variant": self.variant,
            "label_space": self.label_space,
            "n_clips": self.n_clips,
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "per_class_f1": self.per_class_f1,
            "per_class_support": self.per_class_support,
            "confusion_counts": self.confusion_counts,
            "notes": self.notes,
        }


def _score_one_space(
    probs_per_clip: np.ndarray,
    clip_order: Sequence[str],
    space: LabelSpace,
    variant: str,
) -> StandardProtocolResult:
    y_pred = restricted_argmax(probs_per_clip, space)

    # ravdess7 needs to filter to clips whose native label is in the 7-set,
    # which it always is. ravdess8 keeps every clip (Calm clips score at most 0).
    y_true = native_labels_for_clips(clip_order, space)

    classes = list(range(len(space.classes)))
    acc = float(accuracy_score(y_true, y_pred))
    f1m = float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))
    f1w = float(f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0))
    f1_per = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)

    per_class_f1 = {space.classes[c]: float(f1_per[c]) for c in classes}
    per_class_support = {
        space.classes[c]: int(np.sum(y_true == c)) for c in classes
    }

    # Confusion as a {gt -> {pred -> count}} sparse dict.
    confusion: Dict[str, Dict[str, int]] = {c: {} for c in space.classes}
    for gt, pr in zip(y_true.tolist(), y_pred.tolist()):
        gt_name, pr_name = space.classes[gt], space.classes[pr]
        confusion[gt_name][pr_name] = confusion[gt_name].get(pr_name, 0) + 1

    notes = ""
    if space.name == "ravdess8":
        notes = (
            "Strict 8-class reading. The training pipeline collapses RAVDESS "
            "Calm into Neutral, so the model cannot natively predict Calm; "
            "every Calm clip counts as a miss in this row. Use ravdess7 for "
            "the apples-to-apples reading."
        )

    return StandardProtocolResult(
        variant=variant,
        label_space=space.name,
        n_clips=int(len(clip_order)),
        accuracy=acc,
        f1_macro=f1m,
        f1_weighted=f1w,
        per_class_f1=per_class_f1,
        per_class_support=per_class_support,
        confusion_counts=confusion,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Public API: single-checkpoint evaluation
# ---------------------------------------------------------------------------

def evaluate_variant_standard(
    checkpoint: str | Path,
    config_path: str | Path,
    dataset: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
    batch_size: Optional[int] = None,
) -> List[StandardProtocolResult]:
    """Evaluate one trained variant's checkpoint under the standard protocol."""
    cfg = OmegaConf.load(config_path)
    device = cfg.train.device if torch.cuda.is_available() else "cpu"

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    variant = ck["variant"]
    fus_cfg = FusionConfig(**ck["fus_cfg"])
    model = build_model(variant, fus_cfg)
    model.load_state_dict(ck["state_dict"])
    model.to(device)

    loader = _build_val_loader(cfg, variant, batch_size=batch_size)
    logits, videonames = _collect_frame_predictions(model, loader, device)

    if dataset is None:
        dataset = _autodetect_dataset(videonames)
    spaces = _select_label_spaces(dataset)

    probs_per_clip, clip_order = aggregate_softmax_per_clip(logits, videonames)
    results = [_score_one_space(probs_per_clip, clip_order, sp, variant) for sp in spaces]

    if output_dir is None:
        output_dir = Path(cfg.output.results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_md(output_dir / "standard_protocol.md", results, dataset)
    _write_json(output_dir / "standard_protocol.json", results, dataset)
    return results


# ---------------------------------------------------------------------------
# Public API: F0 post-hoc late blend evaluation
# ---------------------------------------------------------------------------

def evaluate_f0_standard(
    visual_checkpoint: str | Path,
    audio_checkpoint: str | Path,
    config_path: str | Path,
    dataset: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
    grid: Sequence[float] = tuple(np.linspace(0.0, 1.0, 21)),
    batch_size: Optional[int] = None,
) -> List[StandardProtocolResult]:
    """Grid-search the F0 EXPR blend weight under the standard protocol.

    F0 has no trained fusion checkpoint --- it post-hoc blends the two
    unimodal heads. Under the AffWild2 per-frame protocol the search
    optimises ``w_expr`` for macro-F1; here we instead optimise ``w_expr``
    for the **standard** weighted-F1 of the dataset's primary label space
    (``cremad6`` or ``ravdess7``). VA and AU weights do not affect the
    EXPR clip-level argmax so they are irrelevant here.
    """
    cfg = OmegaConf.load(config_path)
    device = cfg.train.device if torch.cuda.is_available() else "cpu"

    def _load(ckp):
        ck = torch.load(ckp, map_location="cpu", weights_only=False)
        fus_cfg = FusionConfig(**ck["fus_cfg"])
        m = build_model(ck["variant"], fus_cfg)
        m.load_state_dict(ck["state_dict"])
        return m.to(device)

    m_v = _load(visual_checkpoint)
    m_a = _load(audio_checkpoint)
    loader = _build_val_loader(cfg, "visual_only", batch_size=batch_size)

    logits_v, videonames = _collect_frame_predictions(m_v, loader, device)
    logits_a, _ = _collect_frame_predictions(m_a, loader, device)

    # Pre-compute frame-level softmaxes once.
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / ez.sum(axis=1, keepdims=True)

    p_v = _softmax(logits_v)
    p_a = _softmax(logits_a)

    if dataset is None:
        dataset = _autodetect_dataset(videonames)
    spaces = _select_label_spaces(dataset)
    primary = spaces[0]  # cremad6 or ravdess7

    # Grid-search w on the primary label space's weighted-F1, then report
    # all spaces under that w.
    best_w = 0.5
    best_f1w = -1.0
    for w in grid:
        blended = w * p_v + (1.0 - w) * p_a
        probs_per_clip, clip_order = aggregate_softmax_per_clip(
            np.log(np.clip(blended, 1e-12, 1.0)),  # logits-equivalent for argmax
            videonames,
        )
        # aggregate_softmax_per_clip recomputes a softmax internally, but for
        # already-normalised probabilities the relative ordering is preserved.
        res = _score_one_space(probs_per_clip, clip_order, primary, variant="f0_grid")
        if res.f1_weighted > best_f1w:
            best_f1w = res.f1_weighted
            best_w = float(w)

    # Final scoring at best_w under every label space.
    blended = best_w * p_v + (1.0 - best_w) * p_a
    probs_per_clip, clip_order = aggregate_softmax_per_clip(
        np.log(np.clip(blended, 1e-12, 1.0)), videonames,
    )
    results: List[StandardProtocolResult] = []
    for sp in spaces:
        r = _score_one_space(probs_per_clip, clip_order, sp, variant="f0_grid")
        r.notes = (
            (r.notes + " " if r.notes else "")
            + f"F0 best w_expr = {best_w:.3f} (grid optimised on "
            f"{primary.name} weighted-F1)."
        ).strip()
        results.append(r)

    if output_dir is None:
        output_dir = Path(cfg.output.results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_md(output_dir / "standard_protocol_f0.md", results, dataset)
    _write_json(output_dir / "standard_protocol_f0.json", results, dataset)
    return results


# ---------------------------------------------------------------------------
# Public API: batch sweep over results/interim/
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _RunPlan:
    run_dir: Path
    config_path: Path
    checkpoint_path: Path
    label: str          # printable label for the summary table


def _discover_runs(interim_results_dir: Path) -> List[_RunPlan]:
    plans: List[_RunPlan] = []
    for d in sorted(interim_results_dir.iterdir()):
        if not d.is_dir():
            continue
        ckp = d / "best.pt"
        cfg = d / "config.yaml"
        if not ckp.exists() or not cfg.exists():
            continue
        plans.append(_RunPlan(d, cfg, ckp, label=d.name))
    return plans


def _discover_cross_runs(interim_results_dir: Path) -> List[_RunPlan]:
    """Discover zero-shot transfer evals (CREMA-D ckpt -> RAVDESS val).

    The interim sweep stores the cross-dataset configs under
    ``<interim_results_dir>/_cross_cfgs/<name>.yaml`` (where ``<name>``
    typically ends in ``_on_ravdess``) and the result dir
    ``<interim_results_dir>/<name>/`` holds the per-frame ``metrics.md``
    only --- there is no separate trained checkpoint. The training
    checkpoint comes from the source run, identified here by stripping
    the ``_on_<dataset>`` suffix.
    """
    plans: List[_RunPlan] = []
    cross_dir = interim_results_dir / "_cross_cfgs"
    if not cross_dir.is_dir():
        return plans
    for cfg in sorted(cross_dir.glob("*.yaml")):
        name = cfg.stem
        # Strip the trailing '_on_<dataset>' marker to find the source run.
        if "_on_" not in name:
            continue
        source_run = name.rsplit("_on_", 1)[0]
        src_ckp = interim_results_dir / source_run / "best.pt"
        if not src_ckp.exists():
            continue
        out_dir = interim_results_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        plans.append(_RunPlan(out_dir, cfg, src_ckp, label=name))
    return plans


def run_all_standard_protocol(
    interim_results_dir: str | Path = "results/interim",
    output: str | Path = "results/interim/standard_protocol_summary.md",
    f0_visual_run: Optional[str] = "crema_visual_only",
    f0_audio_run: Optional[str] = "crema_audio_only",
    f0_config: Optional[str | Path] = "configs/interim/crema_f0_grid.yaml",
) -> Path:
    """Sweep every ``results/interim/<run>/`` and write one summary file."""
    interim_results_dir = Path(interim_results_dir)
    output = Path(output)

    plans = _discover_runs(interim_results_dir) + _discover_cross_runs(interim_results_dir)
    rows: List[StandardProtocolResult] = []
    failures: List[Tuple[str, str]] = []

    for plan in plans:
        try:
            res = evaluate_variant_standard(
                checkpoint=plan.checkpoint_path,
                config_path=plan.config_path,
                output_dir=plan.run_dir,
            )
            for r in res:
                r.variant = plan.label
                rows.append(r)
            print(f"[ok] {plan.label}")
        except Exception as exc:
            failures.append((plan.label, str(exc)))
            print(f"[skip] {plan.label}: {exc}")

    # F0 (post-hoc late blend) on CREMA-D, if both unimodals exist.
    if f0_visual_run and f0_audio_run and f0_config:
        v_ckp = interim_results_dir / f0_visual_run / "best.pt"
        a_ckp = interim_results_dir / f0_audio_run / "best.pt"
        if v_ckp.exists() and a_ckp.exists():
            try:
                f0_dir = interim_results_dir / "crema_f0_grid"
                f0_dir.mkdir(parents=True, exist_ok=True)
                f0_res = evaluate_f0_standard(
                    visual_checkpoint=v_ckp,
                    audio_checkpoint=a_ckp,
                    config_path=f0_config,
                    output_dir=f0_dir,
                )
                for r in f0_res:
                    r.variant = "crema_f0_grid"
                    rows.append(r)
                print("[ok] crema_f0_grid")
            except Exception as exc:
                failures.append(("crema_f0_grid", str(exc)))
                print(f"[skip] crema_f0_grid: {exc}")

    _write_summary_md(output, rows, failures)
    print(f"[summary] {output}")
    return output


# ---------------------------------------------------------------------------
# Markdown writers
# ---------------------------------------------------------------------------

def _write_md(path: Path, results: Sequence[StandardProtocolResult], dataset: str) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write(
            "# Standard-protocol evaluation\n\n"
            f"**Dataset detected:** `{dataset}`\n\n"
            "Clip-level (mean-softmax aggregation) accuracy and macro / "
            "weighted-F1 over the native CREMA-D / RAVDESS label space. "
            "These numbers are produced by the SAME checkpoints as the "
            "per-frame Aff-Wild2-aligned scorecard (see `metrics.md`); "
            "they are not the result of additional training.\n\n"
        )
        fh.write(
            "| variant | label space | n_clips | accuracy | F1 macro | F1 weighted |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
        )
        for r in results:
            fh.write(
                f"| {r.variant} | {r.label_space} | {r.n_clips} | "
                f"{r.accuracy:.4f} | {r.f1_macro:.4f} | {r.f1_weighted:.4f} |\n"
            )
        for r in results:
            fh.write(f"\n## Per-class F1 ({r.label_space})\n\n")
            fh.write("| class | F1 | support |\n| --- | --- | --- |\n")
            for c, f in r.per_class_f1.items():
                fh.write(f"| {c} | {f:.4f} | {r.per_class_support[c]} |\n")
            if r.notes:
                fh.write(f"\n_Notes:_ {r.notes}\n")


def _write_json(path: Path, results: Sequence[StandardProtocolResult], dataset: str) -> None:
    payload = {
        "dataset": dataset,
        "results": [r.as_dict() for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_summary_md(
    path: Path,
    results: Sequence[StandardProtocolResult],
    failures: Sequence[Tuple[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, List[StandardProtocolResult]] = {}
    for r in results:
        grouped.setdefault(r.label_space, []).append(r)

    with path.open("w", encoding="utf-8") as fh:
        fh.write(
            "# Standard-protocol summary across all interim runs\n\n"
            "Clip-level accuracy and macro / weighted-F1 over the native "
            "CREMA-D / RAVDESS label spaces. These are the *same* "
            "checkpoints as the per-frame Aff-Wild2-aligned table in "
            "`summary_crema.md`; only the evaluation protocol changes.\n\n"
        )
        for space_name in ("cremad6", "ravdess7", "ravdess8"):
            block = grouped.get(space_name, [])
            if not block:
                continue
            fh.write(f"## Label space: `{space_name}`\n\n")
            fh.write(
                "| run | n_clips | accuracy | F1 macro | F1 weighted |\n"
                "| --- | --- | --- | --- | --- |\n"
            )
            for r in sorted(block, key=lambda x: x.variant):
                fh.write(
                    f"| {r.variant} | {r.n_clips} | {r.accuracy:.4f} | "
                    f"{r.f1_macro:.4f} | {r.f1_weighted:.4f} |\n"
                )
            fh.write("\n")
        if failures:
            fh.write("## Skipped runs\n\n")
            fh.write("| run | reason |\n| --- | --- |\n")
            for name, reason in failures:
                short = reason.replace("\n", " ")[:200]
                fh.write(f"| {name} | {short} |\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Re-evaluate trained checkpoints under the standard CREMA-D / "
            "RAVDESS protocol (clip-level accuracy + macro / weighted-F1)."
        ),
    )
    p.add_argument(
        "--mode",
        default="variant",
        choices=["variant", "f0_grid"],
        help="'variant' for F1-F5 / unimodals; 'f0_grid' for post-hoc blend.",
    )
    p.add_argument("--config", help="OmegaConf YAML for the run.")
    p.add_argument("--checkpoint", help="Required for --mode variant.")
    p.add_argument("--visual-checkpoint", help="Required for --mode f0_grid.")
    p.add_argument("--audio-checkpoint", help="Required for --mode f0_grid.")
    p.add_argument(
        "--dataset",
        choices=["cremad", "ravdess"],
        default=None,
        help="Force a dataset; otherwise auto-detected from videonames.",
    )
    p.add_argument("--output-dir", default=None)
    p.add_argument("--batch-size", type=int, default=None)

    # Batch mode.
    p.add_argument(
        "--run-all", action="store_true",
        help="Sweep every results/interim/<run> with a best.pt + config.yaml.",
    )
    p.add_argument(
        "--interim-results-dir", default="results/interim",
        help="Directory containing the interim run folders.",
    )
    p.add_argument(
        "--output", default="results/interim/standard_protocol_summary.md",
        help="Where to write the run-all summary table.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.run_all:
        run_all_standard_protocol(
            interim_results_dir=args.interim_results_dir,
            output=args.output,
        )
        return

    if not args.config:
        raise SystemExit("--config is required unless --run-all is passed")

    if args.mode == "variant":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for --mode variant")
        evaluate_variant_standard(
            checkpoint=args.checkpoint,
            config_path=args.config,
            dataset=args.dataset,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
    else:
        if not (args.visual_checkpoint and args.audio_checkpoint):
            raise SystemExit(
                "--visual-checkpoint and --audio-checkpoint are required for "
                "--mode f0_grid"
            )
        evaluate_f0_standard(
            visual_checkpoint=args.visual_checkpoint,
            audio_checkpoint=args.audio_checkpoint,
            config_path=args.config,
            dataset=args.dataset,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
