# thesis-code

Bachelor's thesis implementation: efficient multi-task bimodal (visual +
audio) emotion recognition on s-Aff-Wild2 (ABAW-7 MTL track), with a
CREMA-D / RAVDESS cross-validation companion. The code reproduces
Savchenko et al., ABAW-7 MTL (Paper A) on the `mbf_va_mtl` /
`enet_b0_8_va_mtl` EmotiEffLib backbones, adds a HuBERT-large /
wav2vec~2.0 audio branch, and evaluates nine fusion variants on top of
the shared three-head MTL: F0–F5 from the canonical taxonomy and three
F6 candidates (F6a Dirichlet ensemble, F6c IACA-gate, F6d
attention-bottleneck) from the recent ABAW and audiovisual-fusion
literature.

The headline result is a paired six-seed audit (visual_only and F4 at
matched seeds {42, 0, 1, 2, 3, 4}, smoothed): on the composite
P_MTL@0.5, the bimodal advantage does **not** survive paired multi-seed
comparison — bootstrap 95% CI on the mean Δ = F4 − visual_only is
strictly negative at both w=5 and w=10. The residual contribution is a
modality-specific CCC_A effect, localised by F3 gate analysis.

The same audit methodology is then verified end-to-end on the EMI
sub-challenge of ABAW-8 (`notebooks/emi_experiments.ipynb`,
`results/emi/`): on a three-modality (face + audio + Whisper text)
benchmark of different structure, the held-out fusion advantage of
`concat_mlp` and `xattn` over the best single modality (`text/mean`)
is +0.016–0.019 mean Pearson, positive on 6/6 paired seeds (Wilcoxon
p = 0.0312, the minimum attainable at N = 6); `gated` is a confirmed
null. An upstream aggregation result drops out: `mean` pooling beats
the published `stats4` baseline on audio by a 3× factor (0.352 vs
0.115), lifting the reproduced late-fusion baseline from ~0.44 to
0.46 without any architectural change.

## Layout

```
thesis-code/
  pyproject.toml          project + dependency pins
  data/                   .gitignored; Aff-Wild2 / CREMA-D / RAVDESS live here
  configs/                YAML configs per experiment
    interim/              CREMA-D / RAVDESS preliminary runs
    aw2_stage1_*.yaml     Aff-Wild2 visual-only, per backbone
    stage2_*.yaml         Aff-Wild2 audio-only (HuBERT / wav2vec)
    stage3_*.yaml         Aff-Wild2 bimodal fusion (F0–F5 + F6a/c/d)
    stage3_f4_xattn_{w1,w3,w10,wav2vec,wd1e3,mbf}.yaml   F4 ablations
    baseline_{crema,ravdess}_{enet,mbf}.yaml       secondary datasets
    # The matched seed pool (seeds 0..4) is NOT one file per seed: each run
    # is a base config plus `--set seed=N` overrides (see Quick start step 5).
  src/
    utils/{metrics,io}.py     CCC, F1, P_MTL; .npz/.pickle caches
    datasets/
      affwild2_mtl.py         annotation parser + Dataset
      bimodal.py              frame- and window-level bimodal Dataset
      label_mapping.py        categorical → VA anchors + CREMA-D / RAVDESS native labels
      interim_prep.py         secondary corpora preparation
      interim_download.py     fetch helpers
    features/
      extract_visual.py             EmotiEffLib backbone → per-video .npz
      extract_audio_wav.py          video → 16 kHz mono WAV (ffmpeg)
      extract_audio_features.py     HuBERT / wav2vec 2.0 → per-video .npz
      align_audio_to_video.py       temporal resample to video framerate
      extract_faces_from_video.py   face detector fallback
    fusion/
      base.py                 shared FusionConfig + MLP heads
      f0_grid.py              grid-search logit blend (no training)
      variants.py             unimodal controls + F1 concat / F2 blend / F3 gate
      f4_xattn.py             cross-modal attention (two directions)
      f5_lmf.py               low-rank multimodal fusion
      f6a_dirichlet.py        post-hoc Dirichlet-weighted ensemble
      f6c_iaca.py             inconsistency-aware gate on top of F4
      f6d_mbt.py              attention-bottleneck (MBT) fusion
    heads/mtl_head.py         3-head linear MTL model + masked losses
    smoothing.py              per-video gaussian filter (EXPR + VA)
    smooth_fusion.py          Paper-A (σ, δ) grid post-hoc on Stage 3
    eval_frame_average_baseline.py   zero-train per-video aggregation
    eval_standard_protocol.py        CREMA-D / RAVDESS standard protocol
    paired_seed_analysis.py   paired bootstrap + Wilcoxon on matched seeds
    train.py       eval.py            stage-1 visual-only harness
    train_fusion.py eval_fusion.py    stage-3 bimodal harness
  notebooks/                end-to-end reproduction pipeline
    interim_00..05.ipynb            CREMA-D / RAVDESS interim sweep
    aw2_00_setup.ipynb              environment + paths
    aw2_01_extract_visual.ipynb     EmotiEffLib feature extraction
    aw2_02_stage1_train_eval.ipynb  three-head MTL reproduction
    aw2_03_extract_audio.ipynb      HuBERT / wav2vec audio features
    aw2_04_align_audio.ipynb        audio resample onto visual frame timeline
    aw2_05_stage3_fusion.ipynb      F0–F5 bimodal sweep
    aw2_06_stage3_ablations.ipynb   F4 ablations + multi-seed
    aw2_07_stage3_extras.ipynb      F6a / F6c / F6d candidates
    emi_experiments.ipynb           EMI (ABAW-8) §A aggregation, §B fusion,
                                    §C example-bootstrap, §D multi-seed audit
  tests/                    pytest unit tests (no data dependency)
  results/interim/*.md      committed summary tables / ablations
  results/emi/*.md          EMI aggregation / fusion / bootstrap / multiseed tables
  results/                  .gitignored except summary markdowns
  cache/                    .gitignored; per-backbone feature caches
```

## Quick start

```bash
pip install -e .[dev]
pytest -x -v tests/
```

## Aff-Wild2 / s-Aff-Wild2 MTL pipeline

The Aff-Wild2 pipeline is backbone-parameterised via YAML. Two
backbones are wired up: `enet_b0_8_va_mtl` (primary reproduction
target) and `mbf_va_mtl` (secondary; smaller / faster).

```bash
# 1. Visual features (run once per backbone; .npz cache is reused).
python -m src.features.extract_visual --model enet_b0_8_va_mtl \
    --in data/affwild2/cropped_aligned \
    --out cache/features/enet_b0_8_va_mtl

# 2. Stage 1: three-head MTL on visual features only.
python -m src.train --config configs/aw2_stage1_enet.yaml

# 3. Stage 2: audio features (HuBERT-large) + alignment.
jupyter lab notebooks/aw2_03_extract_audio.ipynb
jupyter lab notebooks/aw2_04_align_audio.ipynb

# 4. Stage 3: bimodal fusion sweep (F0–F5).
python -m src.train_fusion --config configs/stage3_f4_xattn.yaml
# ... one config per variant; see notebooks/aw2_05_stage3_fusion.ipynb

# 5. Matched seed pool for the paired audit (seeds 0..4; seed 42 = the base
#    runs above). Each run is a base config with seed + run-name + results-dir
#    overridden via --set; the fully-resolved config is saved to
#    results/<run>/config.yaml. F4 uses the corrected wd=1e-3 / 3-epoch protocol.
for s in 0 1 2 3 4; do
    python -m src.train_fusion --config configs/stage3_visual_only.yaml \
        --set seed=$s --set run_name=stage3_visual_only_seed$s \
        --set output.results_dir=results/stage3_visual_only_seed$s \
        --set train.num_workers=0
    python -m src.train_fusion --config configs/stage3_f4_xattn_wd1e3.yaml \
        --set seed=$s --set run_name=stage3_f4_xattn_seed$s \
        --set output.results_dir=results/stage3_f4_xattn_seed$s
    python -m src.train_fusion --config configs/stage3_f4_xattn_w10_wd1e3.yaml \
        --set seed=$s --set run_name=stage3_f4_xattn_w10_seed$s \
        --set output.results_dir=results/stage3_f4_xattn_w10_seed$s
done

# 6. Post-hoc Paper-A Gaussian smoothing grid per checkpoint.
python -m src.smooth_fusion \
    --checkpoint results/stage3_f4_xattn/best.pt \
    --config configs/stage3_f4_xattn.yaml

# 7. Paired-seed audit: bootstrap CI + Wilcoxon on smoothed P_MTL@0.5.
python -m src.paired_seed_analysis

# 8. F6 candidates (F6a Dirichlet, F6c IACA, F6d MBT).
jupyter lab notebooks/aw2_07_stage3_extras.ipynb
```

## Stage 1 reproduction targets (Paper A cell 60, "aligned" rows)

| backbone | CCC_V | CCC_A | F1_EXPR | P_MTL |
|---|---|---|---|---|
| **enet_b0_8_va_mtl** (primary) | 0.4433 | 0.3422 | 0.5040 | **1.2896** |
| **mbf_va_mtl** (secondary)     | 0.4503 | 0.2870 | 0.4891 | **1.2264** |

Smoothing adds roughly +0.05–0.11 on P_MTL.

## EMI (ABAW-8) verification pipeline

The EMI (Emotional Mimicry Intensity) sub-challenge of ABAW-8
re-runs the s-Aff-Wild2 paired-seed audit on a benchmark of different
structure: per-clip rather than per-frame annotations, three native
modalities (face / audio / text), and a Whisper-transcript embedding
in place of a derived modality. Pre-extracted features from the
HSEmotion ABAW-8 submission are the input: MobileViT-VA face (768-D),
HuBERT-large audio (1024-D), Whisper → text-embedding-3-small text
(1536-D). The pipeline is single-notebook; no extraction step lives
in `src/`.

```bash
# Drop pickles + splits into data/emi/ (see notebook §0 for paths)
#   data/emi/labels/{train_split,valid_split}.csv
#   data/emi/features/emi_mobilevit_va_mtl_orig_faces.pickle
#   data/emi/features/emi_dict_hubert.pickle
#   data/emi/features/emi_whisper_openai_small.pickle
jupyter lab notebooks/emi_experiments.ipynb
```

`emi_experiments.ipynb` is generated from inline cell definitions by
`notebooks/_build_emi_nb.py` (re-run it after editing cells to rebuild the
notebook in place). `notebooks/run_audio_attention.py` is the server-side
CLI that computes the one `audio/attention` aggregation cell on the
supervisor's machine, where the ~17 GB HuBERT feature pickle lives.

Four sections, each persisted under `results/emi/`:

```
§A  agg_table.md                 mean / stats4 / attention pool per modality
§B  fusion_table.md              late_grid / late_learn / concat_mlp / gated / xattn
§C  stat_significance_table.md   N=4588 example-level bootstrap (B=1000), paired p vs text/mean
§D  multiseed_table.md           K=6 seeds × {concat_mlp, gated, xattn},
    multiseed.json               paired bootstrap (B=10 000) + Wilcoxon on per-seed Δ
```

Headline numbers (val mean Pearson, $N_{\mathrm{val}} = 4588$):

| | `text/mean` (ref) | `concat_mlp` | `xattn` | `gated` |
|---|---:|---:|---:|---:|
| single seed | 0.4295 | 0.4469 | 0.4486 | 0.4270 |
| 6-seed mean ± s.d. | 0.4294 ± 0.0021 | 0.4455 ± 0.0010 | 0.4441 ± 0.0046 | 0.4316 ± 0.0047 |
| Δ vs ref, Wilcoxon p | — | **+0.0161**, p=0.0312 (6/6) | **+0.0148**, p=0.0312 (6/6) | +0.0023, p=0.44 (3/6) |

Upstream aggregation finding (Table §A): on audio, `mean` (0.352)
beats `stats4` (0.115) by a 3× factor, lifting the reproduced
late-fusion baseline from 0.44 to 0.46 without any architectural
change.

The `attention` pool over per-frame HuBERT features is missing
from §A because the ~15 GB resident feature dict exceeds the 16 GB
local RAM budget; deferred to a larger machine.

## CREMA-D / RAVDESS interim sweep

The CREMA-D / RAVDESS pipeline is notebook-driven and runs without an
Aff-Wild2 access form:

```bash
jupyter lab notebooks/
# interim_00 → 05 in order
```

The summary tables committed under `results/interim/` are regenerated
by the notebooks and cited from the thesis:

```
results/interim/summary_crema.md            fusion leaderboard on CREMA-D
results/interim/ablation_backbone.md        MBF vs enet visual backbone
results/interim/ablation_cross_dataset.md   CREMA-D → RAVDESS transfer
results/interim/ablation_smoothing.md       per-video smoothing effect
results/interim/ablations_summary.md        consolidated view
results/interim/standard_protocol_summary.md  clip-level external-protocol eval
```

## Caches and reproducibility

Data, feature caches, training checkpoints, and per-run logs are all
`.gitignored`. The only outputs that travel with the repo are the
markdown summary tables under `results/interim/` and the per-run
`smoothing.md` / `smoothing.json` dumps that the thesis tables cite.
Every result row in the thesis corresponds to a `configs/` YAML (optionally
plus `--set` overrides for the seed sweep) and a single `results/<run>/`
directory. Because `train_from_config` writes the fully-resolved config to
`results/<run>/config.yaml`, each run remains exactly reproducible even
when its inputs were a base config plus overrides rather than a standalone
file.
