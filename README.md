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
    stage3_*_seed{0..4}.yaml   matched seed pool for paired audit
    stage3_f4_xattn_{w1,w3,w10,wav2vec,mbf}.yaml   F4 ablations
    baseline_{crema,ravdess}_{enet,mbf}.yaml       secondary datasets
  src/
    utils/{metrics,io}.py     CCC, F1, P_MTL; .npz/.pickle caches
    datasets/
      affwild2_mtl.py         annotation parser + Dataset
      bimodal.py              frame- and window-level bimodal Dataset
      emotion_va_mapping.py   categorical → circumplex VA anchors
      native_labels.py        CREMA-D / RAVDESS adapter to MTL schema
      interim_prep.py         secondary corpora preparation
      interim_download.py     fetch helpers
    features/
      extract_visual.py             EmotiEffLib backbone → per-video .npz
      extract_audio_wav.py          video → 16 kHz mono WAV (ffmpeg)
      extract_audio_features.py     HuBERT / wav2vec 2.0 → per-video .npz
      align_audio_to_video.py       temporal resample to video framerate
      extract_faces_from_video.py   face detector fallback
      build_fps_lookup.py           per-video framerate cache
    fusion/
      base.py                 shared FusionConfig + MLP heads
      f0_grid.py              grid-search logit blend (no training)
      f1_concat.py            early concat
      f2_blend.py             learned scalar blend
      f3_gate.py              per-task sigmoid gate
      f4_xattn.py             cross-modal attention (two directions)
      f5_lmf.py               low-rank multimodal fusion
      f6a_dirichlet.py        post-hoc Dirichlet-weighted ensemble
      f6c_iaca.py             inconsistency-aware gate on top of F4
      f6d_mbt.py              attention-bottleneck (MBT) fusion
      unimodal.py             visual-only / audio-only controls
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
  tests/                    pytest unit tests (no data dependency)
  results/interim/*.md      committed summary tables / ablations
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
python train.py --config configs/aw2_stage1_enet.yaml

# 3. Stage 2: audio features (HuBERT-large) + alignment.
jupyter lab notebooks/aw2_03_extract_audio.ipynb
jupyter lab notebooks/aw2_04_align_audio.ipynb

# 4. Stage 3: bimodal fusion sweep (F0–F5).
python train_fusion.py --config configs/stage3_f4_xattn.yaml
# ... one config per variant; see notebooks/aw2_05_stage3_fusion.ipynb

# 5. F4 ablations + multi-seed (visual_only + F4 at seeds 0..4 for w∈{5,10}).
for cfg in configs/stage3_{visual_only,f4_xattn,f4_xattn_w10}_seed{0..4}.yaml; do
    python train_fusion.py --config "$cfg"
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
Every result row in the thesis corresponds to a YAML config in
`configs/` and a single `results/<run>/` directory.
