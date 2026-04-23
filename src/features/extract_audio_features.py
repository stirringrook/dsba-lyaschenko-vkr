"""Per-video audio feature extraction with a frozen self-supervised encoder.

Mirrors cells 24-26 of ``bah.ipynb``. Produces one ``.npz`` per video:

    features: (T_audio, D_a)   float32
    hop_sec:  ()                float32 (audio-side frame period)
    wav_seconds: ()             float32 (duration of the waveform)

with ``D_a = 768`` for ``wav2vec2-base-960h`` and ``D_a = 1024`` for
``hubert-large-ls960-ft``. Long clips are strided through the encoder in
10-second windows with a 1-second overlap to stay under GPU memory;
overlapping frames are averaged.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm


# ---- model registry -------------------------------------------------------

AUDIO_MODEL_REGISTRY = {
    # short-name -> (hf model id, processor class, model class, feature_dim)
    "wav2vec2_base": ("facebook/wav2vec2-base-960h", "Wav2Vec2Processor", "Wav2Vec2Model", 768),
    "hubert_large": ("facebook/hubert-large-ls960-ft", "AutoProcessor", "HubertModel", 1024),
}


def _load_backbone(short_name: str, device: str):
    """Instantiate one of the registered HF encoders in eval mode."""
    import torch  # noqa: F401  (required by HF)
    import transformers

    if short_name not in AUDIO_MODEL_REGISTRY:
        raise KeyError(
            f"Unknown audio backbone '{short_name}'. "
            f"Available: {list(AUDIO_MODEL_REGISTRY)}"
        )
    hf_id, proc_cls_name, model_cls_name, dim = AUDIO_MODEL_REGISTRY[short_name]
    proc_cls = getattr(transformers, proc_cls_name)
    model_cls = getattr(transformers, model_cls_name)
    processor = proc_cls.from_pretrained(hf_id)
    model = model_cls.from_pretrained(hf_id).to(device)
    model.eval()
    return processor, model, dim


# ---- waveform -> features -------------------------------------------------


def _load_wav(wav_path: Path) -> Tuple[np.ndarray, int]:
    """Load a wav file as ``(waveform[float32], sample_rate)``. Mono."""
    # Deferred import: torchaudio is heavy and not needed for unit tests.
    import torchaudio
    waveform, sr = torchaudio.load(str(wav_path))
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0).numpy().astype(np.float32), int(sr)


def encode_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    processor,
    model,
    device: str,
    window_sec: float = 10.0,
    overlap_sec: float = 1.0,
) -> np.ndarray:
    """Run the frozen encoder over ``waveform`` with overlap-averaged windows.

    Returns shape ``(T_audio, D_a)`` float32.
    """
    import torch

    if waveform.ndim != 1:
        raise ValueError(f"Expected 1-D waveform, got shape {waveform.shape}")
    if len(waveform) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    win = int(window_sec * sample_rate)
    step = max(1, int((window_sec - overlap_sec) * sample_rate))

    # Collect per-window outputs; pad short final window with zeros before encoding.
    outputs = []   # list of (start_sample, np.ndarray [T_w, D])
    feat_per_sample: Optional[float] = None  # audio-frame-per-sample rate

    pos = 0
    while True:
        chunk = waveform[pos : pos + win]
        if len(chunk) == 0:
            break
        if len(chunk) < 400:
            # encoders drop short inputs; pad up so we still produce something.
            chunk = np.concatenate([chunk, np.zeros(400 - len(chunk), dtype=np.float32)])
        inputs = processor(
            chunk, sampling_rate=sample_rate, return_tensors="pt", padding=True
        )
        input_values = inputs["input_values"].to(device)
        with torch.no_grad():
            out = model(input_values)
        h = out.last_hidden_state[0].cpu().numpy()  # (T_w, D)
        outputs.append((pos, h))

        # Record the sample-per-frame ratio once; HF encoders produce ~50 Hz for 16 kHz.
        if feat_per_sample is None:
            feat_per_sample = len(chunk) / max(1, h.shape[0])

        if pos + win >= len(waveform):
            break
        pos += step

    if not outputs:
        return np.zeros((0, 0), dtype=np.float32)

    feat_per_sample = feat_per_sample or (sample_rate / 50.0)
    total_audio_frames = int(np.ceil(len(waveform) / feat_per_sample))
    dim = outputs[0][1].shape[1]
    acc = np.zeros((total_audio_frames, dim), dtype=np.float64)
    counts = np.zeros(total_audio_frames, dtype=np.int32)

    for start_sample, h in outputs:
        start_frame = int(round(start_sample / feat_per_sample))
        end_frame = min(total_audio_frames, start_frame + h.shape[0])
        span = end_frame - start_frame
        acc[start_frame:end_frame] += h[:span]
        counts[start_frame:end_frame] += 1

    counts = np.maximum(counts, 1)
    features = (acc / counts[:, None]).astype(np.float32)
    return features


def extract_audio_features(
    backbone: str,
    wav_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    device: str | None = None,
    window_sec: float = 10.0,
    overlap_sec: float = 1.0,
    overwrite: bool = False,
) -> None:
    """Encode every ``<videoname>.wav`` under ``wav_dir`` with ``backbone``.

    Writes ``output_dir/<videoname>.npz`` with keys ``features, hop_sec,
    wav_seconds``.
    """
    import torch

    wav_dir = Path(wav_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model, _dim = _load_backbone(backbone, device=device)

    wavs = sorted(wav_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(
            f"No .wav files under {wav_dir}. Run extract_audio_wav first."
        )

    for wav_path in tqdm(wavs, desc=f"audio[{backbone}]"):
        out_npz = output_dir / f"{wav_path.stem}.npz"
        if out_npz.exists() and not overwrite:
            continue
        waveform, sr = _load_wav(wav_path)
        if waveform.size == 0:
            continue
        features = encode_waveform(
            waveform=waveform,
            sample_rate=sr,
            processor=processor,
            model=model,
            device=device,
            window_sec=window_sec,
            overlap_sec=overlap_sec,
        )
        wav_seconds = len(waveform) / float(sr)
        hop_sec = wav_seconds / max(1, features.shape[0])
        np.savez(
            out_npz,
            features=features,
            hop_sec=np.float32(hop_sec),
            wav_seconds=np.float32(wav_seconds),
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract frozen audio-encoder features.")
    p.add_argument(
        "--backbone",
        required=True,
        choices=list(AUDIO_MODEL_REGISTRY),
        help="Short name of the audio backbone.",
    )
    p.add_argument("--in", dest="wav_dir", required=True)
    p.add_argument("--out", dest="output_dir", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--window-sec", type=float, default=10.0)
    p.add_argument("--overlap-sec", type=float, default=1.0)
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    extract_audio_features(
        backbone=args.backbone,
        wav_dir=args.wav_dir,
        output_dir=args.output_dir,
        device=args.device,
        window_sec=args.window_sec,
        overlap_sec=args.overlap_sec,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
