"""Per-video face cropping for the interim CREMA-D / RAVDESS pipeline.

Neither CREMA-D nor RAVDESS ships pre-cropped-and-aligned face JPEGs
(unlike AffWild2's ``cropped_aligned/`` layout that the existing
:mod:`src.features.extract_visual` expects). This module closes the gap:
given a source video, it samples frames at a target rate, runs a face
detector on each sampled frame, crops the largest face, resizes it to
``112x112``, and writes the crops as
``<output_dir>/<videoname>/<idx:05d>.jpg``. The resulting per-video
folder drops straight into ``extract_visual.extract_visual_features``.

The detector is OpenCV's built-in Haar cascade
(``haarcascade_frontalface_default.xml``). It ships with ``opencv-python``
so we add no extra dependencies, and it is accurate enough for the
single-actor, frontal, studio-lit clips CREMA-D and RAVDESS contain. A
previous iteration used MediaPipe, but MediaPipe >=0.11 dropped the
``mp.solutions.*`` legacy API and the ``tasks`` replacement needs an
out-of-band ``.tflite`` download.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

from tqdm import tqdm


@dataclass
class FaceCropConfig:
    """Runtime options for :func:`extract_faces_for_clip`."""

    output_size: int = 112                      # match EmotiEffLib backbones.
    target_fps: float = 5.0                     # one crop every 200 ms.
    expand_ratio: float = 0.30                  # pad the detected bbox by 30%.
    scale_factor: float = 1.1                   # Haar image-pyramid step.
    min_neighbors: int = 5                      # Haar false-positive filter.
    min_face_size: int = 60                     # pixels; smallest face to keep.
    image_ext: str = ".jpg"


def _pad_bbox_px(
    x: int, y: int, w: int, h: int, expand: float,
    img_w: int, img_h: int,
) -> Tuple[int, int, int, int]:
    """Pad a pixel bbox ``(x, y, w, h)`` by ``expand`` and clamp to image.

    Returns ``(x0, y0, x1, y1)`` with ``x0 < x1`` and ``y0 < y1`` always.
    """
    dx = int(round(w * expand * 0.5))
    dy = int(round(h * expand * 0.5))
    x0 = max(0, x - dx)
    y0 = max(0, y - dy)
    x1 = min(img_w, x + w + dx)
    y1 = min(img_h, y + h + dy)
    if x1 <= x0:
        x1 = min(img_w, x0 + 1)
    if y1 <= y0:
        y1 = min(img_h, y0 + 1)
    return x0, y0, x1, y1


def _sample_indices(n_frames: int, source_fps: float, target_fps: float) -> List[int]:
    """Return equally spaced frame indices to keep, never exceeding ``n_frames``."""
    if n_frames <= 0:
        return []
    target_fps = max(0.1, float(target_fps))
    source_fps = max(0.1, float(source_fps))
    step = max(1, int(round(source_fps / target_fps)))
    return list(range(0, n_frames, step))


def _parse_rational(text: str) -> float:
    """Parse an ffprobe ``num/den`` rational into a float (0.0 on failure)."""
    if not text or text == "0/0":
        return 0.0
    if "/" in text:
        num, den = text.split("/", 1)
        try:
            n, d = float(num), float(den)
            return n / d if d else 0.0
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _ffprobe_video(video_path: Path) -> Tuple[float, int, int, int]:
    """Probe a video for (fps, n_frames, width, height) via ffprobe.

    Falls back to ``duration * fps`` for containers (FLV included) that
    don't report ``nb_frames`` directly.
    """
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found on PATH; required for FLV fallback")
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate,nb_frames,width,height,duration:format=duration",
            "-of", "json", str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(out.stdout)
    stream = (info.get("streams") or [{}])[0]
    fps = _parse_rational(stream.get("avg_frame_rate", "0/0")) \
        or _parse_rational(stream.get("r_frame_rate", "0/0")) \
        or 30.0
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    n_frames = int(stream.get("nb_frames") or 0)
    if n_frames <= 0:
        duration = float(stream.get("duration") or 0.0)
        if duration <= 0.0:
            duration = float((info.get("format") or {}).get("duration") or 0.0)
        n_frames = int(round(duration * fps))
    return fps, n_frames, width, height


def _iter_frames_ffmpeg(video_path: Path, width: int, height: int) -> Iterator:
    """Stream BGR24 frames out of ffmpeg as numpy arrays.

    Used when ``cv2.VideoCapture`` can't decode the container (e.g. FLV on
    Windows wheels that ship without the FFmpeg backend). The caller is
    responsible for consuming the generator to completion or closing it.
    """
    import numpy as np

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; required for FLV fallback")
    proc = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-nostdin",
            "-i", str(video_path),
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    frame_bytes = width * height * 3
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except Exception:
            proc.kill()


def _crop_and_save(frame_bgr, cascade, cfg: FaceCropConfig, out_sub: Path, written: int) -> int:
    """Run the Haar cascade on one BGR frame and save the best-face crop.

    Returns the updated ``written`` counter (incremented iff a face was
    found and the crop was non-empty).
    """
    import cv2

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=cfg.scale_factor,
        minNeighbors=cfg.min_neighbors,
        minSize=(cfg.min_face_size, cfg.min_face_size),
    )
    if len(faces) == 0:
        return written
    # Largest face by bbox area; detectMultiScale returns (x, y, w, h).
    x, y, w, h = max(faces, key=lambda b: int(b[2]) * int(b[3]))
    x0, y0, x1, y1 = _pad_bbox_px(
        int(x), int(y), int(w), int(h),
        cfg.expand_ratio, frame_bgr.shape[1], frame_bgr.shape[0],
    )
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return written
    resized = cv2.resize(
        crop, (cfg.output_size, cfg.output_size), interpolation=cv2.INTER_AREA,
    )
    # 5-digit zero-padded indices so extract_visual.py sorts in order.
    out_path = out_sub / f"{written:05d}{cfg.image_ext}"
    cv2.imwrite(str(out_path), resized)
    return written + 1


def _load_face_cascade():
    """Load OpenCV's bundled frontal-face Haar cascade."""
    import cv2

    xml = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(xml))
    if cascade.empty():
        raise RuntimeError(
            f"cv2 failed to load Haar cascade at {xml}. Check the "
            f"opencv-python install."
        )
    return cascade


def extract_faces_for_clip(
    video_path: str | Path,
    output_dir: str | Path,
    cfg: Optional[FaceCropConfig] = None,
) -> int:
    """Crop faces from a single video. Returns the number of crops written.

    Tries ``cv2.VideoCapture`` first; if that can't open the file (common
    for ``.flv`` on the Windows ``opencv-python`` wheel, which ships
    without the FFmpeg backend), falls back to streaming raw frames from
    the system ``ffmpeg`` binary.

    The function swallows videos with no detectable face on any sampled
    frame by returning ``0`` --- callers decide how to handle them. It
    raises for videos that neither backend can decode.

    Args:
        video_path: Source video file (mp4, flv, ...).
        output_dir: Destination; a subdirectory named after the video's stem
            is created under here and receives ``00000.jpg, 00001.jpg, ...``.
        cfg: Runtime options; defaults to :class:`FaceCropConfig`.

    Returns:
        Number of frames successfully cropped and saved to disk.
    """
    # Deferred import: opencv-python is heavy and not needed at unit-test
    # time for the parser-only modules.
    import cv2

    cfg = cfg or FaceCropConfig()
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    out_sub = output_dir / video_path.stem
    out_sub.mkdir(parents=True, exist_ok=True)

    cascade = _load_face_cascade()

    cap = cv2.VideoCapture(str(video_path))
    cv2_ok = cap.isOpened()
    if cv2_ok:
        # Some builds open the file but then fail on the first read; probe once.
        probe_ok, probe_frame = cap.read()
        if not probe_ok or probe_frame is None:
            cv2_ok = False
            cap.release()
        else:
            # Rewind -- we'll re-read from frame 0 in the main loop.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    written = 0
    if cv2_ok:
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        keep = set(_sample_indices(n_frames, source_fps, cfg.target_fps))
        try:
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx in keep:
                    written = _crop_and_save(frame, cascade, cfg, out_sub, written)
                frame_idx += 1
        finally:
            cap.release()
    else:
        cap.release()
        fps, n_frames, width, height = _ffprobe_video(video_path)
        if width <= 0 or height <= 0 or n_frames <= 0:
            raise IOError(
                f"cv2 could not open {video_path} and ffprobe returned "
                f"unusable metadata (fps={fps}, n={n_frames}, {width}x{height})"
            )
        keep = set(_sample_indices(n_frames, fps, cfg.target_fps))
        frame_idx = 0
        max_idx = max(keep) if keep else -1
        for frame in _iter_frames_ffmpeg(video_path, width, height):
            if frame_idx in keep:
                written = _crop_and_save(frame, cascade, cfg, out_sub, written)
            frame_idx += 1
            if frame_idx > max_idx:
                break  # stop the ffmpeg pipe early once sampling is done.

    return written


def extract_faces_for_clips(
    video_paths: Iterable[str | Path],
    output_dir: str | Path,
    cfg: Optional[FaceCropConfig] = None,
    skip_existing: bool = True,
) -> dict:
    """Run :func:`extract_faces_for_clip` over many videos, returning counts.

    Args:
        video_paths: Iterable of source video paths.
        output_dir: Destination for the per-video crop folders.
        cfg: Optional :class:`FaceCropConfig`.
        skip_existing: If ``True``, videos whose ``output_dir/<stem>/``
            already contains at least one JPEG are skipped.

    Returns:
        Dict mapping ``videoname -> number_of_crops_written``. Videos with
        no detectable face yield ``0`` (and an empty directory).
    """
    output_dir = Path(output_dir)
    counts: dict = {}
    for vp in tqdm(list(video_paths), desc="faces"):
        vp = Path(vp)
        dest = output_dir / vp.stem
        if skip_existing and dest.is_dir() and any(dest.glob("*.jpg")):
            counts[vp.stem] = len(list(dest.glob("*.jpg")))
            continue
        counts[vp.stem] = extract_faces_for_clip(vp, output_dir, cfg=cfg)
    return counts


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MediaPipe-based face cropping for CREMA-D / RAVDESS."
    )
    p.add_argument("--in", dest="video_dir", required=True,
                   help="Directory containing source video files.")
    p.add_argument("--out", dest="output_dir", required=True,
                   help="Destination for <videoname>/<idx>.jpg crops.")
    p.add_argument("--ext", default=".mp4",
                   help="Source video extension (e.g. .mp4, .flv).")
    p.add_argument("--target-fps", type=float, default=5.0)
    p.add_argument("--output-size", type=int, default=112)
    p.add_argument("--expand-ratio", type=float, default=0.30)
    p.add_argument("--recursive", action="store_true",
                   help="Recurse into subdirectories of --in.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    video_dir = Path(args.video_dir)
    globber = video_dir.rglob if args.recursive else video_dir.glob
    paths = sorted(globber(f"*{args.ext}"))
    cfg = FaceCropConfig(
        output_size=args.output_size,
        target_fps=args.target_fps,
        expand_ratio=args.expand_ratio,
    )
    counts = extract_faces_for_clips(paths, args.output_dir, cfg=cfg)
    missing = sum(1 for v in counts.values() if v == 0)
    print(f"[faces] wrote crops for {len(counts) - missing}/{len(counts)} videos; "
          f"{missing} videos had no detectable face.")


if __name__ == "__main__":
    main()
