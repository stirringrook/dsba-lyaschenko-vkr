"""Build emi_experiments.ipynb from inline cell definitions.

Run from anywhere:
    python thesis-code/notebooks/_build_emi_nb.py

Edit the CELLS list, then re-run to regenerate the notebook in place.
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "emi_experiments.ipynb"


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src,
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


CELLS: list[dict] = []

# -----------------------------------------------------------------------------
CELLS.append(md(
    """# EMI (ABAW8) experiments — aggregation, fusion, statistical significance

Single notebook reproducing the team's `emi.ipynb` baseline in PyTorch and adding three contributions for the VKR §:

- **A — Aggregation.** Frame-level features per video are collapsed to a fixed vector. Team baseline uses concat(mean, std, min, max) (`stats4`). We compare `mean`, `stats4`, and a learned `attention` pool on each of three modalities (face / audio / text).
- **B — Fusion.** Single-modality predictions are combined. Compared variants: late grid-search (team baseline), learnable concat-MLP, gated, and cross-attention over modality tokens.
- **C — Statistical significance.** For every variant in A and B, 1000-resample bootstrap on the validation set produces a 95 % CI on the mean-Pearson metric. A paired bootstrap against the best single-modality baseline produces a two-sided p-value per variant.

All features are pre-extracted server-side — see `emi_server_and_features.md` at the repo root for what to copy and where. The notebook does not download anything; it fails fast in §0 if a required pickle is missing.

Outputs (saved to `thesis-code/results/emi/`):

- `agg_table.md` — §A results.
- `fusion_table.md` — §B results.
- `stat_significance_table.md` — §C results.
- `summary.json` — all numbers + per-variant val predictions for paste into LaTeX.
"""
))

# -----------------------------------------------------------------------------
CELLS.append(md("""## 0. Setup and file presence"""))

CELLS.append(code(
    """import json, pickle, time, hashlib
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import stats

REPO = Path.cwd().resolve()
if REPO.name == 'notebooks':
    REPO = REPO.parent
DATA       = REPO / 'data' / 'emi'
LABELS_DIR = DATA / 'labels'
FEAT_DIR   = DATA / 'features'
RESULTS    = REPO / 'results' / 'emi'
RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

EMOTIONS = ['Admiration', 'Amusement', 'Determination', 'Empathic Pain', 'Excitement', 'Joy']
NUM_TARGETS = len(EMOTIONS)

print('Repo       :', REPO)
print('Device     :', DEVICE)
print('Features at:', FEAT_DIR, '(exists:', FEAT_DIR.exists(), ')')
print('Results to :', RESULTS)
"""
))

CELLS.append(code(
    """REQUIRED = {
    'face'     : FEAT_DIR / 'emi_mobilevit_va_mtl_orig_faces.pickle',
    'audio'    : FEAT_DIR / 'emi_dict_hubert.pickle',
    'text'     : FEAT_DIR / 'emi_whisper_openai_small.pickle',
    'train_csv': LABELS_DIR / 'train_split.csv',
    'val_csv'  : LABELS_DIR / 'valid_split.csv',
}
missing = {k: str(v) for k, v in REQUIRED.items() if not v.exists()}
if missing:
    msg = '\\n'.join(f'  {k:9s}: {v}' for k, v in missing.items())
    raise FileNotFoundError(
        'Missing files. See emi_server_and_features.md \\u00a73 for scp commands.\\n' + msg
    )
print('All required files present.')
"""
))

# -----------------------------------------------------------------------------
CELLS.append(md("""## 1. Loaders

The three feature pickles have distinct layouts (confirmed by reading the team notebook):

- Face: `[video2feat, video2scores]` — we keep only `video2feat`.
- Audio: `dict[str, np.ndarray]`.
- Text: `dict[str, np.ndarray]` with shape either `[T_chunks, D]` or `[D]` — we promote the 1-D case to `[1, D]`.

Labels CSVs have a header and six target columns in the canonical order.
"""))

CELLS.append(code(
    """def load_face_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, (list, tuple)) and len(obj) == 2 and isinstance(obj[0], dict):
        video2feat, _ = obj
    else:
        video2feat = obj
    return {k: np.asarray(v, dtype=np.float32) for k, v in video2feat.items()}

def load_audio_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return {k: np.asarray(v, dtype=np.float32) for k, v in obj.items()}

def load_text_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    out = {}
    for k, v in obj.items():
        a = np.asarray(v, dtype=np.float32)
        if a.ndim == 1:
            a = a[None, :]
        out[k] = a
    return out

def load_labels(path):
    df = pd.read_csv(path, dtype={0: str})
    name_col = df.columns[0]
    target_cols = list(df.columns[1:])
    assert len(target_cols) == NUM_TARGETS, (
        f'expected {NUM_TARGETS} target cols, got {len(target_cols)}: {target_cols}'
    )
    out = {}
    for _, row in df.iterrows():
        out[str(row[name_col])] = row[target_cols].to_numpy(dtype=np.float32)
    return out

face_feats   = load_face_pickle(REQUIRED['face'])
audio_feats  = load_audio_pickle(REQUIRED['audio'])
text_feats   = load_text_pickle(REQUIRED['text'])
train_labels = load_labels(REQUIRED['train_csv'])
val_labels   = load_labels(REQUIRED['val_csv'])

def _sample_shape(d):
    return next(iter(d.values())).shape

print(f'face : {len(face_feats):5d} videos, sample shape {_sample_shape(face_feats)}')
print(f'audio: {len(audio_feats):5d} videos, sample shape {_sample_shape(audio_feats)}')
print(f'text : {len(text_feats):5d} videos, sample shape {_sample_shape(text_feats)}')
print(f'train: {len(train_labels)} labels, val: {len(val_labels)} labels')
"""
))

CELLS.append(code(
    """# Intersected video lists ensure §A, §B and §C compare the same examples.
NAMES_TR = sorted(set(train_labels) & set(face_feats) & set(audio_feats) & set(text_feats))
NAMES_VA = sorted(set(val_labels)   & set(face_feats) & set(audio_feats) & set(text_feats))
print(f'Train intersection: {len(NAMES_TR)} / {len(train_labels)} ({len(NAMES_TR)/len(train_labels):.1%})')
print(f'Val   intersection: {len(NAMES_VA)} / {len(val_labels)}   ({len(NAMES_VA)/len(val_labels):.1%})')

Y_TR = np.stack([train_labels[n] for n in NAMES_TR]).astype(np.float32)
Y_VA = np.stack([val_labels[n]   for n in NAMES_VA]).astype(np.float32)
print('Y_TR', Y_TR.shape, 'Y_VA', Y_VA.shape)
"""
))

# -----------------------------------------------------------------------------
CELLS.append(md("""## 2. Metrics, loss, MLP head, attention pool, trainers

`mean_pearson` is what the EMI challenge reports. `pearson_loss` is the same training objective the team used (`1 - mean Pearson` across the six classes, computed batch-wise).

Two trainers:
- `train_static(X_tr, y_tr, X_va, y_va)` — for any pre-aggregated feature matrix.
- `train_attn(video2feat, ...)` — keeps frame-level features and learns an attention pooling weight per frame.
"""))

CELLS.append(code(
    """def mean_pearson(preds, labels):
    \"\"\"Mean Pearson r over the 6 emotion columns. Also returns per-class array.\"\"\"
    per = np.array([stats.pearsonr(preds[:, i], labels[:, i])[0] for i in range(preds.shape[1])])
    return float(per.mean()), per

def pearson_loss(pred, tgt):
    eps = 1e-8
    p = pred - pred.mean(0, keepdim=True)
    t = tgt - tgt.mean(0, keepdim=True)
    num = (p * t).sum(0)
    den = torch.sqrt((p * p).sum(0) * (t * t).sum(0)) + eps
    return 1.0 - (num / den).mean()


class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden=128, n_layers=1, out=NUM_TARGETS, dropout=0.5):
        super().__init__()
        layers, prev = [], in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden
        layers.append(nn.Linear(prev, out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_static(X_tr, y_tr, X_va, y_va, *,
                 hidden=128, n_layers=1, dropout=0.5,
                 lr=1e-3, weight_decay=1e-4,
                 epochs=80, batch_size=64, patience=15, seed=SEED, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLPHead(X_tr.shape[1], hidden=hidden, n_layers=n_layers, dropout=dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    X_tr_t = torch.from_numpy(X_tr).to(DEVICE)
    y_tr_t = torch.from_numpy(y_tr).to(DEVICE)
    X_va_t = torch.from_numpy(X_va).to(DEVICE)
    best_p, best_preds, bad = -np.inf, None, 0
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(len(X_tr))
        for i in range(0, len(idx), batch_size):
            j = idx[i:i + batch_size]
            opt.zero_grad()
            loss = pearson_loss(model(X_tr_t[j]), y_tr_t[j])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            preds = model(X_va_t).cpu().numpy()
        m, _ = mean_pearson(preds, y_va)
        if m > best_p:
            best_p, best_preds, bad = m, preds, 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 10 == 0:
            print(f'  ep{ep:3d} val={m:.4f}  best={best_p:.4f}')
    return best_preds, best_p


class AttnPoolHead(nn.Module):
    def __init__(self, in_dim, hidden=128, n_layers=1, out=NUM_TARGETS, dropout=0.5):
        super().__init__()
        self.attn = nn.Linear(in_dim, 1)
        layers, prev = [], in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden
        layers.append(nn.Linear(prev, out))
        self.head = nn.Sequential(*layers)

    def forward(self, x, mask):
        scores = self.attn(x).squeeze(-1)
        scores = scores.masked_fill(~mask, float('-inf'))
        w = scores.softmax(dim=1).unsqueeze(-1)
        pooled = (w * x).sum(dim=1)
        return self.head(pooled)


class FrameDataset(Dataset):
    def __init__(self, video2feat, names, label_map):
        self.items = [(n, label_map[n], video2feat[n]) for n in names]
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        n, y, x = self.items[i]
        return torch.from_numpy(x), torch.from_numpy(y), n


def pad_collate(batch):
    xs, ys, names = zip(*batch)
    T = max(x.shape[0] for x in xs)
    D = xs[0].shape[1]
    X = torch.zeros(len(xs), T, D)
    mask = torch.zeros(len(xs), T, dtype=torch.bool)
    for i, x in enumerate(xs):
        X[i, :x.shape[0]] = x
        mask[i, :x.shape[0]] = True
    return X, mask, torch.stack(ys), names


def train_attn(video2feat, names_tr, names_va, *,
               hidden=128, n_layers=1, dropout=0.5,
               lr=1e-3, weight_decay=1e-4,
               epochs=80, batch_size=16, patience=15, seed=SEED, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_ds = FrameDataset(video2feat, names_tr, train_labels)
    va_ds = FrameDataset(video2feat, names_va, val_labels)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  collate_fn=pad_collate)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)
    in_dim = next(iter(video2feat.values())).shape[1]
    model = AttnPoolHead(in_dim, hidden=hidden, n_layers=n_layers, dropout=dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_p, best_preds, bad = -np.inf, None, 0
    y_va_canon = np.stack([val_labels[n] for n in names_va]).astype(np.float32)
    for ep in range(epochs):
        model.train()
        for X, mask, Y, _ in tr_dl:
            X, mask, Y = X.to(DEVICE), mask.to(DEVICE), Y.to(DEVICE)
            opt.zero_grad()
            loss = pearson_loss(model(X, mask), Y)
            loss.backward()
            opt.step()
        model.eval()
        preds_all = []
        with torch.no_grad():
            for X, mask, _, _ in va_dl:
                X, mask = X.to(DEVICE), mask.to(DEVICE)
                preds_all.append(model(X, mask).cpu().numpy())
        preds = np.concatenate(preds_all)
        m, _ = mean_pearson(preds, y_va_canon)
        if m > best_p:
            best_p, best_preds, bad = m, preds, 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 10 == 0:
            print(f'  ep{ep:3d} val={m:.4f}  best={best_p:.4f}')
    return best_preds, best_p
"""
))

# -----------------------------------------------------------------------------
CELLS.append(md("""## §A — Aggregation comparison

For each modality, sweep three pooling strategies:
- `mean`  — single mean vector over time.
- `stats4`— concat(mean, std, min, max). Team baseline.
- `attention` — learned attention pool (frame weights from a 1-D linear over each frame).

For the static aggregations we materialise `(X, y)` matrices and call `train_static`. For `attention` we keep frame-level tensors and call `train_attn`.
"""))

CELLS.append(code(
    """def agg_mean(x):
    return x.mean(axis=0)

def agg_stats4(x):
    return np.concatenate([x.mean(0), x.std(0), x.min(0), x.max(0)])

AGG_STATIC = {'mean': agg_mean, 'stats4': agg_stats4}

def build_matrix(video2feat, names, label_map, agg_fn):
    X = np.stack([agg_fn(video2feat[n]) for n in names]).astype(np.float32)
    y = np.stack([label_map[n]          for n in names]).astype(np.float32)
    return X, y

MODALITIES = {
    'face' : face_feats,
    'audio': audio_feats,
    'text' : text_feats,
}

agg_results = []            # list of dicts: modality, agg, mean_p, per_class, val_preds (np), in_dim
val_preds_by_variant = {}   # variant_key -> np.ndarray [N_val, 6]
in_dim_by_modality   = {}

for mname, mfeat in MODALITIES.items():
    in_dim_by_modality[mname] = next(iter(mfeat.values())).shape[1]
    for agg_name, fn in AGG_STATIC.items():
        X_tr, y_tr = build_matrix(mfeat, NAMES_TR, train_labels, fn)
        X_va, y_va = build_matrix(mfeat, NAMES_VA, val_labels,   fn)
        t0 = time.time()
        preds, best_p = train_static(X_tr, y_tr, X_va, y_va, hidden=128, n_layers=1)
        per = np.array([stats.pearsonr(preds[:, i], y_va[:, i])[0] for i in range(NUM_TARGETS)])
        dt = time.time() - t0
        key = f'{mname}/{agg_name}'
        val_preds_by_variant[key] = preds
        agg_results.append({
            'modality': mname, 'agg': agg_name, 'in_dim': X_tr.shape[1],
            'mean_pearson': best_p, 'per_class': per.tolist(),
            'time_s': round(dt, 1),
        })
        print(f'{key:18s}  D={X_tr.shape[1]:5d}  mean_P={best_p:.4f}  ({dt:.1f}s)')

    # attention pool — learned, frame-level
    t0 = time.time()
    preds_a, best_p_a = train_attn(mfeat, NAMES_TR, NAMES_VA, hidden=128, n_layers=1)
    per_a = np.array([stats.pearsonr(preds_a[:, i], Y_VA[:, i])[0] for i in range(NUM_TARGETS)])
    dt = time.time() - t0
    key = f'{mname}/attention'
    val_preds_by_variant[key] = preds_a
    agg_results.append({
        'modality': mname, 'agg': 'attention', 'in_dim': in_dim_by_modality[mname],
        'mean_pearson': best_p_a, 'per_class': per_a.tolist(),
        'time_s': round(dt, 1),
    })
    print(f'{key:18s}  D={in_dim_by_modality[mname]:5d}  mean_P={best_p_a:.4f}  ({dt:.1f}s)')
"""
))

CELLS.append(code(
    """# Best aggregation per modality (used as the single-modality input to §B).
best_per_modality = {}
for mname in MODALITIES:
    rows = [r for r in agg_results if r['modality'] == mname]
    best = max(rows, key=lambda r: r['mean_pearson'])
    best_per_modality[mname] = best
    print(f"best for {mname:5s}: agg={best['agg']:9s}  mean_P={best['mean_pearson']:.4f}")
"""
))

CELLS.append(code(
    """# Render §A table.
lines = ['# EMI §A — aggregation comparison (val mean Pearson)\\n']
lines.append('| Modality | Aggregation | in_dim | mean Pearson | Admiration | Amusement | Determination | Empathic Pain | Excitement | Joy |')
lines.append('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
for r in agg_results:
    per = r['per_class']
    lines.append(
        f"| {r['modality']} | {r['agg']} | {r['in_dim']} | **{r['mean_pearson']:.4f}** | "
        + ' | '.join(f'{v:.3f}' for v in per) + ' |'
    )
agg_md = '\\n'.join(lines) + '\\n'
(RESULTS / 'agg_table.md').write_text(agg_md, encoding='utf-8')
print(agg_md)
"""
))

# -----------------------------------------------------------------------------
CELLS.append(md("""## §B — Fusion comparison

All variants are evaluated on the same intersected val set so paired bootstrap in §C is well-defined. The single-modality inputs use the **best aggregation per modality** from §A.

Variants:

- **late_grid** — post-hoc per-emotion weight search over `{face, audio, text}` predictions in steps of 0.05. Matches the team's baseline procedure but without their `bias` term (we found it's within noise).
- **late_learn** — same idea, but the per-emotion 3-vector of weights is a learnable parameter optimised against val (no train). Sanity check that grid resolution isn't the bottleneck.
- **concat_mlp** — concatenate the `stats4` vectors of all three modalities, train an MLP head end-to-end.
- **gated** — project each modality to a shared dim, learn a softmax gate over modalities per sample.
- **xattn** — three modality tokens (`stats4` of each, projected to shared dim), one transformer-encoder block, mean-pool tokens, MLP head.
"""))

CELLS.append(code(
    """# Per-modality stats4 train/val matrices, used by every learned fusion variant.
def stats4_matrix(video2feat, names):
    return np.stack([agg_stats4(video2feat[n]) for n in names]).astype(np.float32)

X_TR = {m: stats4_matrix(MODALITIES[m], NAMES_TR) for m in MODALITIES}
X_VA = {m: stats4_matrix(MODALITIES[m], NAMES_VA) for m in MODALITIES}
print({m: (X_TR[m].shape, X_VA[m].shape) for m in MODALITIES})
"""
))

CELLS.append(code(
    """# Best single-modality val predictions (from §A best_per_modality choice).
single_preds = {m: val_preds_by_variant[f\"{m}/{best_per_modality[m]['agg']}\"] for m in MODALITIES}

fusion_results = []

# --- late_grid ---
def late_grid(single_preds, y_va, step=0.05):
    \"\"\"Per-emotion 3-weight grid search on val.\"\"\"
    mods = list(single_preds.keys())
    P = np.stack([single_preds[m] for m in mods], axis=-1)  # [N, 6, M]
    weights = np.zeros((NUM_TARGETS, len(mods)))
    grid = np.arange(0.0, 1.0 + 1e-6, step)
    for i in range(NUM_TARGETS):
        best_r, best_w = -np.inf, None
        for wa in grid:
            for wb in grid:
                wc = 1.0 - wa - wb
                if wc < -1e-6 or wc > 1.0 + 1e-6:
                    continue
                w = np.array([wa, wb, max(0.0, wc)])
                pred = (P[:, i, :] * w).sum(axis=-1)
                r = stats.pearsonr(pred, y_va[:, i])[0]
                if r > best_r:
                    best_r, best_w = r, w
        weights[i] = best_w
    # Build full predictions matrix.
    full = np.zeros_like(y_va)
    for i in range(NUM_TARGETS):
        full[:, i] = (P[:, i, :] * weights[i]).sum(axis=-1)
    return full, weights

preds_lg, w_lg = late_grid(single_preds, Y_VA)
m_lg, per_lg = mean_pearson(preds_lg, Y_VA)
val_preds_by_variant['fusion/late_grid'] = preds_lg
fusion_results.append({'variant': 'late_grid', 'mean_pearson': m_lg, 'per_class': per_lg.tolist(),
                       'extra': {'weights': w_lg.tolist()}})
print(f'late_grid    mean_P={m_lg:.4f}')
"""
))

CELLS.append(code(
    """# --- late_learn: optimise per-emotion 3-vector weights against val Pearson directly ---
def late_learn(single_preds, y_va, steps=2000, lr=0.05, seed=SEED):
    torch.manual_seed(seed)
    mods = list(single_preds.keys())
    P = torch.from_numpy(np.stack([single_preds[m] for m in mods], axis=-1)).to(DEVICE)  # [N, 6, M]
    Y = torch.from_numpy(y_va).to(DEVICE)
    raw = torch.zeros(NUM_TARGETS, len(mods), device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr)
    for _ in range(steps):
        w = raw.softmax(dim=-1)
        pred = (P * w.unsqueeze(0)).sum(dim=-1)
        opt.zero_grad()
        pearson_loss(pred, Y).backward()
        opt.step()
    with torch.no_grad():
        w = raw.softmax(dim=-1)
        pred = (P * w.unsqueeze(0)).sum(dim=-1).cpu().numpy()
    return pred, w.cpu().numpy()

preds_ll, w_ll = late_learn(single_preds, Y_VA)
m_ll, per_ll = mean_pearson(preds_ll, Y_VA)
val_preds_by_variant['fusion/late_learn'] = preds_ll
fusion_results.append({'variant': 'late_learn', 'mean_pearson': m_ll, 'per_class': per_ll.tolist(),
                       'extra': {'weights': w_ll.tolist()}})
print(f'late_learn   mean_P={m_ll:.4f}')
"""
))

CELLS.append(code(
    """# --- concat_mlp: concat(stats4 of three modalities) -> MLPHead ---
def concat_mlp_run(seed=SEED, hidden=256, n_layers=1, dropout=0.5):
    X_tr_c = np.concatenate([X_TR[m] for m in MODALITIES], axis=1)
    X_va_c = np.concatenate([X_VA[m] for m in MODALITIES], axis=1)
    return train_static(X_tr_c, Y_TR, X_va_c, Y_VA,
                        hidden=hidden, n_layers=n_layers, dropout=dropout, seed=seed)

preds_c, m_c = concat_mlp_run()
per_c = np.array([stats.pearsonr(preds_c[:, i], Y_VA[:, i])[0] for i in range(NUM_TARGETS)])
val_preds_by_variant['fusion/concat_mlp'] = preds_c
fusion_results.append({'variant': 'concat_mlp', 'mean_pearson': m_c, 'per_class': per_c.tolist(), 'extra': {}})
print(f'concat_mlp   mean_P={m_c:.4f}')
"""
))

CELLS.append(code(
    """# --- gated: per-modality projection, softmax gate over modalities (per sample) -> head ---
class GatedFusion(nn.Module):
    def __init__(self, dims, shared=128, hidden=128, out=NUM_TARGETS, dropout=0.5):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d, shared) for d in dims])
        self.gate = nn.Linear(sum(dims), len(dims))
        self.head = nn.Sequential(
            nn.Linear(shared, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out),
        )
    def forward(self, xs):
        z = torch.stack([F.relu(self.proj[i](xs[i])) for i in range(len(xs))], dim=1)  # [B, M, S]
        g = self.gate(torch.cat(xs, dim=1)).softmax(dim=-1).unsqueeze(-1)               # [B, M, 1]
        fused = (g * z).sum(dim=1)                                                      # [B, S]
        return self.head(fused), g.squeeze(-1)


def train_gated(seed=SEED, hidden=128, shared=128, dropout=0.5,
                lr=1e-3, weight_decay=1e-4, epochs=80, batch_size=64, patience=15):
    torch.manual_seed(seed); np.random.seed(seed)
    dims = [X_TR[m].shape[1] for m in MODALITIES]
    model = GatedFusion(dims, shared=shared, hidden=hidden, dropout=dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    xs_tr = [torch.from_numpy(X_TR[m]).to(DEVICE) for m in MODALITIES]
    xs_va = [torch.from_numpy(X_VA[m]).to(DEVICE) for m in MODALITIES]
    y_tr_t = torch.from_numpy(Y_TR).to(DEVICE)
    best_p, best_preds, best_g, bad = -np.inf, None, None, 0
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(len(Y_TR))
        for i in range(0, len(idx), batch_size):
            j = idx[i:i + batch_size]
            opt.zero_grad()
            out, _ = model([x[j] for x in xs_tr])
            pearson_loss(out, y_tr_t[j]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            preds, g = model(xs_va)
            preds = preds.cpu().numpy()
            g = g.cpu().numpy()
        m, _ = mean_pearson(preds, Y_VA)
        if m > best_p:
            best_p, best_preds, best_g, bad = m, preds, g, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best_preds, best_p, best_g

preds_g, m_g, gates = train_gated()
per_g = np.array([stats.pearsonr(preds_g[:, i], Y_VA[:, i])[0] for i in range(NUM_TARGETS)])
val_preds_by_variant['fusion/gated'] = preds_g
fusion_results.append({'variant': 'gated', 'mean_pearson': m_g, 'per_class': per_g.tolist(),
                       'extra': {'mean_gate_per_modality': gates.mean(axis=0).tolist(),
                                 'modality_order': list(MODALITIES)}})
print(f'gated        mean_P={m_g:.4f}  mean gates={dict(zip(MODALITIES, gates.mean(axis=0).round(3)))}')
"""
))

CELLS.append(code(
    """# --- xattn: three modality tokens, one transformer-encoder block, mean-pool, MLP head ---
class XAttnFusion(nn.Module):
    def __init__(self, dims, shared=128, n_heads=4, ff=256, hidden=128, out=NUM_TARGETS, dropout=0.3):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d, shared) for d in dims])
        self.pos = nn.Parameter(torch.randn(1, len(dims), shared) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model=shared, nhead=n_heads, dim_feedforward=ff,
                                         dropout=dropout, batch_first=True, activation='gelu')
        self.block = nn.TransformerEncoder(enc, num_layers=1)
        self.head = nn.Sequential(
            nn.Linear(shared, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out),
        )
    def forward(self, xs):
        tokens = torch.stack([F.relu(self.proj[i](xs[i])) for i in range(len(xs))], dim=1)  # [B, M, S]
        tokens = tokens + self.pos
        tokens = self.block(tokens)
        return self.head(tokens.mean(dim=1))


def train_xattn(seed=SEED, shared=128, n_heads=4, ff=256, hidden=128, dropout=0.3,
                lr=5e-4, weight_decay=1e-4, epochs=80, batch_size=64, patience=15):
    torch.manual_seed(seed); np.random.seed(seed)
    dims = [X_TR[m].shape[1] for m in MODALITIES]
    model = XAttnFusion(dims, shared=shared, n_heads=n_heads, ff=ff,
                        hidden=hidden, dropout=dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    xs_tr = [torch.from_numpy(X_TR[m]).to(DEVICE) for m in MODALITIES]
    xs_va = [torch.from_numpy(X_VA[m]).to(DEVICE) for m in MODALITIES]
    y_tr_t = torch.from_numpy(Y_TR).to(DEVICE)
    best_p, best_preds, bad = -np.inf, None, 0
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(len(Y_TR))
        for i in range(0, len(idx), batch_size):
            j = idx[i:i + batch_size]
            opt.zero_grad()
            out = model([x[j] for x in xs_tr])
            pearson_loss(out, y_tr_t[j]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            preds = model(xs_va).cpu().numpy()
        m, _ = mean_pearson(preds, Y_VA)
        if m > best_p:
            best_p, best_preds, bad = m, preds, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best_preds, best_p

preds_x, m_x = train_xattn()
per_x = np.array([stats.pearsonr(preds_x[:, i], Y_VA[:, i])[0] for i in range(NUM_TARGETS)])
val_preds_by_variant['fusion/xattn'] = preds_x
fusion_results.append({'variant': 'xattn', 'mean_pearson': m_x, 'per_class': per_x.tolist(), 'extra': {}})
print(f'xattn        mean_P={m_x:.4f}')
"""
))

CELLS.append(code(
    """# Render §B table.
lines = ['# EMI §B — fusion comparison (val mean Pearson)\\n']
lines.append('| Variant | mean Pearson | Admiration | Amusement | Determination | Empathic Pain | Excitement | Joy |')
lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
for m in MODALITIES:
    r = best_per_modality[m]
    lines.append(f"| {m}/{r['agg']} (best single) | **{r['mean_pearson']:.4f}** | "
                 + ' | '.join(f'{v:.3f}' for v in r['per_class']) + ' |')
for r in fusion_results:
    lines.append(f"| {r['variant']} | **{r['mean_pearson']:.4f}** | "
                 + ' | '.join(f'{v:.3f}' for v in r['per_class']) + ' |')
fusion_md = '\\n'.join(lines) + '\\n'
(RESULTS / 'fusion_table.md').write_text(fusion_md, encoding='utf-8')
print(fusion_md)
"""
))

# -----------------------------------------------------------------------------
CELLS.append(md("""## §C — Statistical significance

For every variant we compute:

1. **Bootstrap 95 % CI** on val mean Pearson — resample the N validation videos with replacement 1000 times, recompute mean Pearson on each resample, report the 2.5 / 97.5 percentiles. Same resample seed for every variant so the rows are directly comparable.
2. **Paired bootstrap p-value vs. reference.** For each variant, draw 1000 paired resamples (same indices applied to both the variant and the reference baseline), compute the delta Δ = variant − reference per resample, and report a two-sided p-value `2·min(P(Δ≤0), P(Δ≥0))`. Reference: the **best single-modality variant from §A**.

This is the same construction we used for the Aff-Wild2 paired-seed analysis (`src/paired_seed_analysis.py`), just done across val examples instead of seeds.
"""))

CELLS.append(code(
    """N_BOOT = 1000

def bootstrap_pearson_ci(preds, y_va, idx_matrix):
    \"\"\"Mean and 2.5/97.5-percentile bootstrap CI on mean Pearson.\"\"\"
    boots = np.empty(idx_matrix.shape[0])
    for b, idx in enumerate(idx_matrix):
        p_b = preds[idx]
        y_b = y_va[idx]
        per = np.array([stats.pearsonr(p_b[:, i], y_b[:, i])[0] for i in range(NUM_TARGETS)])
        boots[b] = per.mean()
    point = float(np.array([stats.pearsonr(preds[:, i], y_va[:, i])[0] for i in range(NUM_TARGETS)]).mean())
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), boots


def paired_bootstrap(preds_a, preds_b, y_va, idx_matrix):
    \"\"\"Two-sided p-value for H0: mean Pearson of A equals mean Pearson of B (paired by example).\"\"\"
    deltas = np.empty(idx_matrix.shape[0])
    for b, idx in enumerate(idx_matrix):
        y_b = y_va[idx]
        pa = preds_a[idx]
        pb = preds_b[idx]
        ma = np.array([stats.pearsonr(pa[:, i], y_b[:, i])[0] for i in range(NUM_TARGETS)]).mean()
        mb = np.array([stats.pearsonr(pb[:, i], y_b[:, i])[0] for i in range(NUM_TARGETS)]).mean()
        deltas[b] = ma - mb
    # Centre deltas at zero (Hall and Wilson 1991 style for two-sided p-value).
    centred = deltas - deltas.mean()
    observed = deltas.mean()
    p = 2.0 * min(np.mean(centred >= abs(observed)), np.mean(centred <= -abs(observed)))
    return float(observed), float(p), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)), deltas


rng = np.random.default_rng(SEED)
N = Y_VA.shape[0]
IDX = rng.integers(0, N, size=(N_BOOT, N))
print(f'val size N={N}, bootstrap resamples={N_BOOT}')
"""
))

CELLS.append(code(
    """# Reference: best single-modality variant from §A.
ref_row = max([r for r in agg_results], key=lambda r: r['mean_pearson'])
ref_key = f\"{ref_row['modality']}/{ref_row['agg']}\"
ref_preds = val_preds_by_variant[ref_key]
print(f'Reference: {ref_key}  point={ref_row[\"mean_pearson\"]:.4f}')

stat_rows = []
variant_keys = list(val_preds_by_variant.keys())
t0 = time.time()
for k in variant_keys:
    point, lo, hi, _ = bootstrap_pearson_ci(val_preds_by_variant[k], Y_VA, IDX)
    if k == ref_key:
        delta, pval, dlo, dhi = 0.0, 1.0, 0.0, 0.0
    else:
        delta, pval, dlo, dhi, _ = paired_bootstrap(
            val_preds_by_variant[k], ref_preds, Y_VA, IDX,
        )
    stat_rows.append({
        'variant'      : k,
        'mean_pearson' : point,
        'ci_lo'        : lo, 'ci_hi': hi,
        'delta_vs_ref' : delta,
        'delta_lo'     : dlo, 'delta_hi': dhi,
        'p_value'      : pval,
    })
    print(f'{k:24s}  P={point:.4f}  CI=[{lo:.4f},{hi:.4f}]  d={delta:+.4f}  p={pval:.4f}')
print(f'\\nbootstrap took {time.time()-t0:.1f}s')
"""
))

CELLS.append(code(
    """# Render §C table — sorted by point estimate, descending.
stat_rows_sorted = sorted(stat_rows, key=lambda r: -r['mean_pearson'])
lines = [f'# EMI §C — bootstrap 95 % CI and paired p-value (N={N}, B={N_BOOT})\\n']
lines.append(f'Reference for paired bootstrap: `{ref_key}` (best single-modality variant).\\n')
lines.append('| Variant | mean Pearson | 95 % CI | Δ vs ref | 95 % CI on Δ | p (two-sided) |')
lines.append('| --- | ---: | --- | ---: | --- | ---: |')
for r in stat_rows_sorted:
    ci   = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
    dci  = f"[{r['delta_lo']:+.3f}, {r['delta_hi']:+.3f}]"
    sig  = '' if r['variant'] == ref_key else ('  ✱' if r['p_value'] < 0.05 else '')
    lines.append(f"| `{r['variant']}`{sig} | **{r['mean_pearson']:.4f}** | {ci} | {r['delta_vs_ref']:+.4f} | {dci} | {r['p_value']:.4f} |")
lines.append('')
lines.append('✱ = significant at α=0.05 (paired bootstrap, two-sided, not multiple-testing corrected).')
stat_md = '\\n'.join(lines) + '\\n'
(RESULTS / 'stat_significance_table.md').write_text(stat_md, encoding='utf-8')
print(stat_md)
"""
))

# -----------------------------------------------------------------------------
CELLS.append(md("""## Persist everything for the VKR

`summary.json` contains the raw numbers and the per-variant val predictions (as lists), enough to redo any plot or run a follow-up analysis without re-training.
"""))

CELLS.append(code(
    """summary = {
    'config': {
        'seed': SEED,
        'device': str(DEVICE),
        'n_val': int(N),
        'n_train': int(len(NAMES_TR)),
        'n_bootstrap': N_BOOT,
        'emotions': EMOTIONS,
        'feature_files': {k: str(v) for k, v in REQUIRED.items()},
        'reference_for_paired_bootstrap': ref_key,
    },
    'agg_results'    : agg_results,
    'best_per_modality': {m: {'agg': r['agg'], 'mean_pearson': r['mean_pearson']}
                          for m, r in best_per_modality.items()},
    'fusion_results' : fusion_results,
    'stat_rows'      : stat_rows,
    'val_preds': {k: v.tolist() for k, v in val_preds_by_variant.items()},
    'y_val'    : Y_VA.tolist(),
    'names_val': NAMES_VA,
}
(RESULTS / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print('wrote', RESULTS / 'summary.json')
print('wrote', RESULTS / 'agg_table.md')
print('wrote', RESULTS / 'fusion_table.md')
print('wrote', RESULTS / 'stat_significance_table.md')
"""
))

CELLS.append(md("""---

### Where to take this in §6 / §7 of the report

- **§A** answers the supervisor's prompt directly: drop in the row for `audio/attention` vs `audio/stats4` — if attention pool wins, the *aggregation* gives a small, defensible audio-side improvement; if not, you've explicitly tested it and can leave the team baseline in place with a justified citation.
- **§B** gives the cross-modal fusion contribution. Pick one as the headline (likely `xattn` or `gated`) and report all four for honesty. The `late_grid` row reproduces the team baseline and pins your numbers to theirs.
- **§C** is what makes A and B publishable. Quote `mean ± CI` (not bare numbers) in the table, and call out `(p = <value>)` only when ≤0.05. The asterisks in `stat_significance_table.md` mark those rows.

The full reference numbers from `emi.ipynb` (face 0.18, audio 0.30, text 0.40, late grid 0.44) are listed in `emi_server_and_features.md` §7 for spot-comparison.
"""))

# -----------------------------------------------------------------------------
notebook = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print("wrote", OUT, f"({len(CELLS)} cells)")
