"""Fill the audio/attention cell of EMI tab:emi_agg.

Runs only the missing cell: HuBERT per-frame features, learned attention pool,
seed 42, the same hyperparameters used for every other (modality, aggregation)
cell in the table. Designed for the supervisor's server where the 17 GB pickle
exists at FEAT_PATH and 29 GB of RAM is available.

Usage on server:
    CUDA_VISIBLE_DEVICES=0 python3 run_audio_attention.py \
        --feat   /home/HDD6TB/avsavchenko/src/emotions-multimodal/faces/ABAW/abaw8/emi_dict_hubert.pickle \
        --train  /home/HDD6TB/datasets/emotions/ABAW/ABAW_8/EMI/train_split.csv \
        --val    /home/HDD6TB/datasets/emotions/ABAW/ABAW_8/EMI/valid_split.csv \
        --out    audio_attention_result.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import stats

EMOTIONS = ['Admiration', 'Amusement', 'Determination', 'Empathic Pain', 'Excitement', 'Joy']
NUM_TARGETS = len(EMOTIONS)


def mean_pearson(preds: np.ndarray, labels: np.ndarray):
    per = np.array([stats.pearsonr(preds[:, i], labels[:, i])[0] for i in range(preds.shape[1])])
    return float(per.mean()), per


def pearson_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    p = pred - pred.mean(0, keepdim=True)
    t = tgt - tgt.mean(0, keepdim=True)
    num = (p * t).sum(0)
    den = torch.sqrt((p * p).sum(0) * (t * t).sum(0) + eps)
    return 1.0 - (num / den).mean()


class AttnPoolHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, n_layers: int = 1,
                 out: int = NUM_TARGETS, dropout: float = 0.5):
        super().__init__()
        self.attn = nn.Linear(in_dim, 1)
        layers, prev = [], in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden
        layers.append(nn.Linear(prev, out))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attn(x).squeeze(-1)
        scores = scores - scores.amax(dim=1, keepdim=True).detach()
        exp_scores = scores.exp() * mask.float()
        w = exp_scores / (exp_scores.sum(dim=1, keepdim=True) + 1e-12)
        w = w.unsqueeze(-1)
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


def load_audio_pickle(path: Path, t_cap: int | None = None, min_t: int = 1):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    out = {}
    n_capped, n_dropped = 0, 0
    for k, v in obj.items():
        a = np.asarray(v, dtype=np.float32)
        if a.shape[0] < min_t:
            n_dropped += 1
            continue
        if t_cap is not None and a.shape[0] > t_cap:
            idx = np.linspace(0, a.shape[0] - 1, t_cap).round().astype(np.int64)
            a = a[idx]
            n_capped += 1
        out[k] = a
    print(f'  uniform-subsampled to T<={t_cap}: {n_capped} videos; '
          f'dropped T<{min_t}: {n_dropped} videos', flush=True)
    return out


def load_labels(path: Path):
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


def compute_frame_stats(video2feat, names):
    """Per-dimension mean and std over all training frames."""
    total = None
    total_sq = None
    n_frames = 0
    for n in names:
        x = video2feat[n].astype(np.float64)
        if total is None:
            total = x.sum(axis=0)
            total_sq = (x * x).sum(axis=0)
        else:
            total += x.sum(axis=0)
            total_sq += (x * x).sum(axis=0)
        n_frames += x.shape[0]
    mean = total / n_frames
    var = total_sq / n_frames - mean * mean
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def zscore_inplace(video2feat, mean, std):
    inv_std = 1.0 / std
    neg_mean_over_std = -mean * inv_std
    for k, v in video2feat.items():
        video2feat[k] = (v * inv_std + neg_mean_over_std).astype(np.float32)


def train_attn(video2feat, names_tr, names_va, train_labels, val_labels, *,
               device, hidden=128, n_layers=1, dropout=0.5,
               lr=1e-3, weight_decay=1e-4,
               epochs=80, batch_size=16, patience=15, seed=42, verbose=True,
               grad_clip=1.0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_ds = FrameDataset(video2feat, names_tr, train_labels)
    va_ds = FrameDataset(video2feat, names_va, val_labels)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  collate_fn=pad_collate)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)

    in_dim = next(iter(video2feat.values())).shape[1]
    model = AttnPoolHead(in_dim, hidden=hidden, n_layers=n_layers, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    y_va_canon = np.stack([val_labels[n] for n in names_va]).astype(np.float32)
    best_p, best_preds, bad, best_ep = -np.inf, None, 0, -1

    for ep in range(epochs):
        model.train()
        for batch_i, (X, mask, Y, names) in enumerate(tr_dl):
            X, mask, Y = X.to(device), mask.to(device), Y.to(device)
            opt.zero_grad()
            out = model(X, mask)
            if not torch.isfinite(out).all():
                raise RuntimeError(
                    f'ep{ep} batch{batch_i} non-finite output. '
                    f'X.shape={tuple(X.shape)}  any_nan_X={torch.isnan(X).any().item()}  '
                    f'mask_rowsums_min={mask.sum(1).min().item()}  '
                    f'names={list(names)}'
                )
            loss = pearson_loss(out, Y)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f'ep{ep} batch{batch_i} non-finite loss={loss.item()}. '
                    f'out_std0_min={out.std(0).min().item():.3e}  '
                    f'names={list(names)}'
                )
            loss.backward()
            any_bad_grad = any(
                (p.grad is not None) and (not torch.isfinite(p.grad).all())
                for p in model.parameters()
            )
            if any_bad_grad:
                opt.zero_grad()
                if verbose:
                    print(f'  ep{ep} batch{batch_i}: non-finite grad, step skipped', flush=True)
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        model.eval()
        preds_all = []
        with torch.no_grad():
            for X, mask, _, _ in va_dl:
                X, mask = X.to(device), mask.to(device)
                preds_all.append(model(X, mask).cpu().numpy())
        preds = np.concatenate(preds_all)
        m, _ = mean_pearson(preds, y_va_canon)
        if m > best_p:
            best_p, best_preds, bad, best_ep = m, preds, 0, ep
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f'  early stop at ep{ep} (best ep{best_ep} val={best_p:.4f})')
                break
        if verbose:
            print(f'  ep{ep:3d} val={m:.4f}  best={best_p:.4f}', flush=True)

    return best_preds, best_p, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat',  required=True, type=Path)
    ap.add_argument('--train', required=True, type=Path)
    ap.add_argument('--val',   required=True, type=Path)
    ap.add_argument('--out',   required=True, type=Path)
    ap.add_argument('--seed',  type=int, default=42)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--t-cap', type=int, default=1024,
                    help='uniform-subsample frame count cap per video (default 1024)')
    ap.add_argument('--min-t', type=int, default=5,
                    help='drop videos with fewer than this many frames (default 5)')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device, flush=True)
    if device.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0), flush=True)

    t0 = time.time()
    print(f'Loading audio pickle (t_cap={args.t_cap}, min_t={args.min_t}) ...', flush=True)
    audio_feats = load_audio_pickle(args.feat, t_cap=args.t_cap, min_t=args.min_t)
    Ts = np.array([v.shape[0] for v in audio_feats.values()])
    print(f'  {len(audio_feats)} videos; T stats: min={Ts.min()}, '
          f'p50={int(np.percentile(Ts,50))}, p99={int(np.percentile(Ts,99))}, '
          f'max={Ts.max()}; load took {time.time() - t0:.1f}s', flush=True)

    print('Loading labels ...', flush=True)
    train_labels = load_labels(args.train)
    val_labels   = load_labels(args.val)
    print(f'  train={len(train_labels)}  val={len(val_labels)}', flush=True)

    NAMES_TR = sorted(set(train_labels) & set(audio_feats))
    NAMES_VA = sorted(set(val_labels)   & set(audio_feats))
    print(f'Intersected: train={len(NAMES_TR)}, val={len(NAMES_VA)}', flush=True)

    print('Computing per-dim train-frame stats for z-score ...', flush=True)
    t_stats = time.time()
    mean, std = compute_frame_stats(audio_feats, NAMES_TR)
    print(f'  mean range [{mean.min():.3f}, {mean.max():.3f}]  '
          f'std range [{std.min():.3f}, {std.max():.3f}]  '
          f'({time.time() - t_stats:.1f}s)', flush=True)
    print('Applying z-score in place ...', flush=True)
    t_zs = time.time()
    zscore_inplace(audio_feats, mean, std)
    print(f'  done ({time.time() - t_zs:.1f}s)', flush=True)

    t_train = time.time()
    preds, best_p, best_ep = train_attn(
        audio_feats, NAMES_TR, NAMES_VA, train_labels, val_labels,
        device=device,
        hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
    )
    train_secs = time.time() - t_train

    y_va = np.stack([val_labels[n] for n in NAMES_VA]).astype(np.float32)
    _, per_class = mean_pearson(preds, y_va)

    result = {
        'variant': 'audio/attention',
        'seed': args.seed,
        'hidden': args.hidden,
        'in_dim': int(next(iter(audio_feats.values())).shape[1]),
        'n_train': len(NAMES_TR),
        'n_val': len(NAMES_VA),
        'best_epoch': int(best_ep),
        'mean_pearson': float(best_p),
        'per_class': {e: float(p) for e, p in zip(EMOTIONS, per_class)},
        'train_secs': round(train_secs, 1),
        'total_secs': round(time.time() - t0, 1),
    }
    args.out.write_text(json.dumps(result, indent=2))
    print('\n=== RESULT ===')
    print(json.dumps(result, indent=2))
    print(f'\nSaved to {args.out}')


if __name__ == '__main__':
    main()
