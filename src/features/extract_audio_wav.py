"""Extract 16 kHz mono ``.wav`` from AffWild2 video files via ffmpeg.

Matches cell 22 of ``bah.ipynb``:

    ffmpeg -y -i <in>.mp4 -ac 1 -ar 16000 -vn <out>.wav

For clips without an audio track we log the videoname and write a
``<name>.silent`` marker file so downstream extraction can skip them
cleanly. The rest of the pipeline treats these as zero-filled audio
features.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List

from tqdm import tqdm


VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
AUDIO_EXTS = (".wav", ".flac", ".m4a", ".aac", ".ogg", ".mp3")
SOURCE_EXTS = VIDEO_EXTS + AUDIO_EXTS


def _find_videos(videos_dir: Path) -> List[Path]:
    """Return every video- or audio-source file under ``videos_dir`` (recursive, sorted)."""
    out: List[Path] = []
    for root, _, files in os.walk(videos_dir):
        for f in files:
            if f.lower().endswith(SOURCE_EXTS):
                out.append(Path(root) / f)
    out.sort()
    return out


def _has_audio_stream(video_path: Path, ffprobe_bin: str = "ffprobe") -> bool:
    """Return True iff ``video_path`` contains at least one audio stream."""
    try:
        res = subprocess.run(
            [
                ffprobe_bin,
                "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "default=nw=1:nk=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return "audio" in res.stdout
    except FileNotFoundError:
        # ffprobe missing: be permissive; ffmpeg will fail loudly later.
        return True


def extract_audio_wavs(
    videos_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    sample_rate: int = 16000,
    channels: int = 1,
    ffmpeg_bin: str = "ffmpeg",
    overwrite: bool = False,
) -> List[Path]:
    """Extract one ``.wav`` per video at ``sample_rate`` Hz, ``channels`` channels.

    Args:
        videos_dir: Directory containing the AffWild2 video files.
        output_dir: Destination directory for ``<videoname>.wav``.
        sample_rate: Target audio sample rate (default 16000, matches bah.ipynb).
        channels: Target audio channel count (default 1 = mono).
        ffmpeg_bin: ffmpeg executable name or path.
        overwrite: Recompute existing ``.wav`` files.

    Returns:
        List of paths to the written ``.wav`` files (excludes skipped silent clips).
    """
    videos_dir = Path(videos_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which(ffmpeg_bin) is None:
        raise FileNotFoundError(
            f"ffmpeg binary '{ffmpeg_bin}' not on PATH. "
            "Install ffmpeg (https://ffmpeg.org) before running."
        )

    videos = _find_videos(videos_dir)
    if not videos:
        raise FileNotFoundError(f"No videos found under {videos_dir}")

    written: List[Path] = []
    silent: List[Path] = []
    for vid in tqdm(videos, desc="ffmpeg"):
        out_wav = output_dir / f"{vid.stem}.wav"
        silent_marker = output_dir / f"{vid.stem}.silent"
        if out_wav.exists() and not overwrite:
            written.append(out_wav)
            continue
        if silent_marker.exists() and not overwrite:
            continue

        if not _has_audio_stream(vid):
            silent_marker.write_text("no audio stream\n", encoding="utf-8")
            silent.append(vid)
            continue

        cmd = [
            ffmpeg_bin, "-y",
            "-i", str(vid),
            "-ac", str(channels),
            "-ar", str(sample_rate),
            "-vn",
            str(out_wav),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            # Some files have broken audio; tolerate and mark silent.
            silent_marker.write_text(
                f"ffmpeg failed:\n{res.stderr[:2000]}\n", encoding="utf-8"
            )
            silent.append(vid)
            if out_wav.exists():
                out_wav.unlink(missing_ok=True)
            continue
        written.append(out_wav)

    if silent:
        print(f"[audio] {len(silent)} clips had no usable audio; marker files written.")
    print(f"[audio] wrote {len(written)} wav files to {output_dir}")
    return written


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ffmpeg: video -> 16 kHz mono wav.")
    p.add_argument("--in", dest="videos_dir", required=True)
    p.add_argument("--out", dest="output_dir", required=True)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    extract_audio_wavs(
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        channels=args.channels,
        ffmpeg_bin=args.ffmpeg,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
