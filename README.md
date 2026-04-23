# thesis-code

Bachelor's thesis implementation: efficient multi-task bimodal (visual +
audio) emotion recognition for ABAW. The code reproduces Savchenko et al.,
ABAW-7 MTL (Paper A) on the `mbf_va_mtl` / `enet_b0_8_va_mtl` EmotiEffLib
backbones, adds a HuBERT-large audio branch, and trains six fusion
variants (F0--F5) on top of the shared three-head MTL.

Because Aff-Wild2 access was still pending at the time of the
pre-thesis internship report, the repository currently targets the
**interim evaluation phase** on CREMA-D and RAVDESS. The Aff-Wild2
protocol is wired up end-to-end and ready to run once the data lands.

## Layout

```
thesis-code/
  pyproject.toml          project + dependency pins
  data/                   .gitignored; CREMA-D, RAVDESS, Aff-Wild2 live here
  configs/                YAML configs per experiment
    interim/              CREMA-D / RAVDESS runs (current phase)
    stage1_*.yaml         Aff-Wild2 visual-only, per backbone
    stage2_*.yaml         Aff-Wild2 audio-only
    stage3_*.yaml         Aff-Wild2 bimodal fusion F0--F5
  src/
    utils/{metrics,io}.py     CCC, F1, P_MTL; .npz/.pickle caches
    datasets/
      affwild2_mtl.py         annotation parser + Dataset
      bimodal.py              frame- and window-level bimodal Dataset
      emotion_va_mapping.py   categorical -> circumplex VA anchors
      interim_prep.py         CREMA-D / RAVDESS -> Aff-Wild2 schema
      interim_download.py     fetch helpers
    features/
      extract_visual.py         EmotiEffLib backbone -> per-video .npz
      extract_audio_wav.py      video -> 16 kHz mono WAV (ffmpeg)
      extract_audio_features.py HuBERT / wav2vec 2.0 -> per-video .npz
      align_audio_to_video.py   temporal resample to video framerate
      extract_faces_from_video.py face detector fallback
    fusion/
      base.py                 shared FusionConfig + MLP heads
      f0_grid.py              grid-search logit blend (no training)
      f1_concat.py            early concat
      f2_blend.py             learned scalar blend
      f3_gate.py              per-task sigmoid gate
      f4_xattn.py             cross-modal attention (two directions)
      f5_lmf.py               low-rank multimodal fusion
      unimodal.py             visual-only / audio-only controls
    heads/mtl_head.py         3-head linear MTL model + masked losses
    smoothing.py              per-video gaussian filter (EXPR + VA)
    train.py       eval.py            stage-1 visual-only harness
    train_fusion.py eval_fusion.py    stage-3 bimodal harness
  notebooks/                end-to-end reproduction pipeline
    interim_00_download.ipynb       fetch CREMA-D + RAVDESS
    interim_01_data_prep.ipynb      faces, wavs, features, annotations
    interim_02_train_variants.ipynb train the 9 variants
    interim_03_ablations.ipynb      backbone + cross-dataset
    interim_04_smoothing.ipynb      smoothing ablation
    interim_05_figures.ipynb        PDFs for the thesis report
  tests/                    pytest unit tests (no data dependency)
  results/interim/*.md      committed summary tables / ablations
  results/                  .gitignored except for the summary markdowns
  cache/                    .gitignored; per-backbone feature caches
```

## Interim phase quick start (CREMA-D + RAVDESS)

The interim pipeline is notebook-driven. Run the notebooks in order
once the environment is set up:

```bash
pip install -e .[dev]

jupyter lab notebooks/
# 00 -> 01 -> 02 -> 03 -> 04 -> 05
```

Each notebook writes a `results/<run_name>/` directory with
`log.csv`, `metrics.md`, and `best.pt`. The summary tables the report
cites are regenerated under `results/interim/`:

```
results/interim/summary_crema.md           fusion leaderboard on CREMA-D
results/interim/ablation_backbone.md       MBF vs enet visual backbone
results/interim/ablation_cross_dataset.md  CREMA-D -> RAVDESS transfer
results/interim/ablation_smoothing.md      per-video smoothing effect
results/interim/ablations_summary.md       consolidated view
```

These six markdown files are the only committed outputs under
`results/`; everything else (checkpoints, training logs) is
`.gitignored`.

## Stage 1 quick start (once Aff-Wild2 is on disk)

The Aff-Wild2 pipeline is backbone-parameterised via YAML. Two
backbones are wired up: `enet_b0_8_va_mtl` (primary reproduction
target) and `mbf_va_mtl` (secondary; smaller / faster).

```bash
# 1. Extract visual features. Run once per backbone; the .npz cache is reused.
python -m src.features.extract_visual \
    --model enet_b0_8_va_mtl \
    --in   data/affwild2/cropped_aligned \
    --out  cache/features/enet_b0_8_va_mtl

python -m src.features.extract_visual \
    --model mbf_va_mtl \
    --in   data/affwild2/cropped_aligned \
    --out  cache/features/mbf_va_mtl

# 2. Train three heads (EXPR -> VA -> AU, sequentially). Pick a config.
python train.py --config configs/stage1_visual_enet.yaml
python train.py --config configs/stage1_visual.yaml        # mbf_va

# 3. Evaluate.
python eval.py --config configs/stage1_visual_enet.yaml \
    --checkpoint results/stage1_visual_enet/best.pt

# 4. Evaluate again with gaussian smoothing (EXPR + VA only, not AU).
python eval.py --config configs/stage1_visual_enet.yaml \
    --checkpoint results/stage1_visual_enet/best.pt \
    --smooth --sigma 100000 --delta 50
```

## Stage 1 reproduction targets (paper A, cell 60, "aligned" rows)

| backbone | ccc_V | ccc_A | F1_EXPR | P_MTL |
|---|---|---|---|---|
| **enet_b0_8_va_mtl** (primary) | 0.4433 | 0.3422 | 0.5040 | **1.2896** |
| **mbf_va_mtl** (secondary)     | 0.4503 | 0.2870 | 0.4891 | **1.2264** |

Smoothing adds roughly +0.05--0.07 on P_MTL.

## Running the tests (no data needed)

```bash
pip install -e .[dev]
pytest -x -v tests/
```
