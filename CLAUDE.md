# Project
Bachelor's thesis on efficient multi-task bimodal (visual + audio) emotion
recognition for ABAW. English. Supervisor: Andrey V. Savchenko.

# Reference code
The EmotiEffLib source and Savchenko's competition notebooks live at
`../EmotiEffLib-main/`. READ them for reference; NEVER modify them. The two
files you read most often are:
  - EmotiEffLib-main/training_and_examples/ABAW/ABAW7/mtl.ipynb   (paper A)
  - EmotiEffLib-main/training_and_examples/ABAW/ABAW8/bah.ipynb   (bimodal)

# Data
AffWild2 lives at ./data/affwild2/. The cropped_aligned/ subdirectory
contains per-video 112x112 face JPEGs. Annotations are under
./data/affwild2/annotations/{training_set_annotations.txt, validation_set_annotations.txt}.
All data is gitignored.

# Environment
- Python 3.12.10 on Windows; CUDA GPU is available.
- Use `bash` syntax when running shell commands (forward slashes, `/dev/null`).

# Conventions
- Python 3.10+, PyTorch. Never TensorFlow (reference notebook uses Keras;
  port it to PyTorch on the way in).
- Features are cached as .npz per video under cache/features/<backbone>/.
- Audio features under cache/features/<audio_backbone>/.
- All configs go under configs/*.yaml; load with OmegaConf.
- Every training run writes a results/<run_name>/ directory with
  log.csv, metrics.md, best.pt.
- Losses MUST mask missing labels using the mask columns (see
  src/datasets/affwild2_mtl.py). Never train on -5 VA rows.

# Stage 1 targets (paper A cell 60, "aligned" rows)
#   backbone            ccc_V   ccc_A   F1_EXPR  P_MTL
#   enet_b0_8_va_mtl    0.4433  0.3422  0.5040   1.2896   (primary)
#   mbf_va_mtl          0.4503  0.2870  0.4891   1.2264   (secondary)
# Smoothing (sigma=100000, delta=50) adds +0.05-0.07 on P_MTL on top.
#
# Input dims X = [features, scores]:
#   enet_b0_8_va_mtl    1280 + 10 = 1290
#   mbf_va_mtl           512 + 10 =  522

# Style
- Write docstrings in Google style.
- Every new function gets a minimal pytest test under tests/.
- Run pytest and fix failures before declaring a session complete.
- Prefer explicit over clever. This is research code we will re-read in 6 months.
