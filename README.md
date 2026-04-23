# thesis-code

Bachelor's thesis implementation: efficient multi-task bimodal (visual + audio)
emotion recognition for ABAW. Stage 1 reproduces Savchenko et al., ABAW-7 MTL
(Paper A) using the `mbf_va_mtl` backbone from EmotiEffLib.

## Layout

```
thesis-code/
  CLAUDE.md               persistent instructions for Claude Code
  pyproject.toml          project + dependency pins
  data/                   .gitignored; AffWild2 MTL lives here
  configs/                YAML configs per experiment
  src/
    utils/metrics.py      CCC, F1, P_MTL
    utils/io.py           .npz/.pickle caches
    datasets/affwild2_mtl.py   annotation parser + Dataset
    features/extract_visual.py MT-EmotiEffLib -> per-video .npz
    heads/mtl_head.py     3-head linear MTL model + losses
    smoothing.py          per-video gaussian filter
    train.py              sequential per-head training
    eval.py               CCC/F1/P_MTL on validation split
  tests/                  pytest unit tests (no data dependency)
  notes/                  reading-guide prose (mtl_notebook.md)
  notebooks/              exploratory
  results/                logs, checkpoints, metrics.md
  cache/features/         per-backbone feature pickles (.npz)
```

## Stage 1 quick start (once AffWild2 is on disk)

The pipeline is backbone-parameterised via YAML. Two backbones are
wired up: `enet_b0_8_va_mtl` (primary reproduction target) and
`mbf_va_mtl` (secondary; smaller / faster).

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

Smoothing adds roughly +0.05-0.07 on P_MTL.

## Running the tests (no data needed)

```bash
pip install -e .[dev]
pytest -x -v tests/
```
