"""Build a per-video FPS lookup for the Aff-Wild2 source videos.

Output: ``data/affwild2/fps_lookup.json``, mapping cache stem ->
{"fps": float, "source_file": str, "source_kind": "filename" | "probe"}.

Why this exists
---------------
The MTL annotation files key every row by ``<videoname>/<frame_idx>``.
Per-video frame counts in the visual cache are already correct (they
came from the same frame stream the annotations index into), so
``align_audio_to_video.py`` does not need an explicit FPS map at
runtime --- it works in fractional units of [0, 1]. We still build
the map for two reasons:

1. Sanity check: roughly half the MTL-release source files have FPS
   embedded in the filename (e.g. ``102-30-640x360.mp4`` -> 30 fps).
   Cross-checking the probed FPS against the filename catches any
   delivered file that is not what its name claims.
2. Audio extraction debugging: if a clip's wav2vec/HuBERT cache has
   far fewer (or more) tokens than ``T_video / fps * 50``, FPS is the
   first thing to check.

Two FPS sources, in priority order:

* Filename-decorated FPS: filenames of the form ``<id>-<fps>-<WxH>``
  or ``<id>-<fps>-<WxH>-<chunk>`` are parsed directly. Cheapest.
* OpenCV / ffprobe probe: for unsuffixed filenames (``102.avi``,
  ``video28.mp4``, ``451.avi``, etc.) we open the file with
  ``cv2.VideoCapture`` and read ``CAP_PROP_FPS``; if that fails we
  fall back to a system ``ffprobe`` call.

Both probe paths require either an ``opencv-python`` install
(bundles its own FFmpeg DLL on Windows) or an ``ffprobe`` binary on
PATH.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

_DECORATED = re.compile(
    r"^(.+?)-(\d+)-(\d+)x(\d+)(?:-(\d+))?(?:_(left|right))?\.(mp4|avi|mov)$",
    re.IGNORECASE,
)


def fps_from_filename(name: str) -> float | None:
    m = _DECORATED.match(name)
    return float(m.group(2)) if m else None


def fps_from_probe(path: Path) -> float | None:
    """Try OpenCV first, then ffprobe."""
    try:
        import cv2  # type: ignore
    except ImportError:
        cv2 = None
    if cv2 is not None:
        cap = cv2.VideoCapture(str(path))
        try:
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps and fps > 0 and fps < 1000:
                    return float(fps)
        finally:
            cap.release()

    if shutil.which("ffprobe"):
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate",
                "-of", "default=nokey=1:noprint_wrappers=1",
                str(path),
            ],
            capture_output=True, text=True, check=False,
        )
        rate = out.stdout.strip()
        if "/" in rate:
            num, den = rate.split("/")
            try:
                num_f = float(num)
                den_f = float(den)
                if den_f != 0:
                    return num_f / den_f
            except ValueError:
                return None
        try:
            return float(rate)
        except ValueError:
            return None
    return None


def resolve_source(cache_stem: str, by_stem: dict[str, list[Path]]) -> Path | None:
    if cache_stem in by_stem:
        return by_stem[cache_stem][0]
    base = cache_stem.replace("_left", "").replace("_right", "")
    if base in by_stem:
        return by_stem[base][0]
    return None


def build(
    videos_root: Path,
    cache_dir: Path | None,
    out_path: Path,
) -> dict:
    by_stem: dict[str, list[Path]] = {}
    for batch in ("batch1", "batch2", "new_vids"):
        d = videos_root / batch
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.suffix.lower() in (".mp4", ".avi", ".mov"):
                by_stem.setdefault(p.stem, []).append(p)

    if cache_dir is not None:
        targets = sorted({p.stem for p in cache_dir.glob("*.npz")})
    else:
        targets = sorted(by_stem.keys())

    lookup: dict[str, dict] = {}
    unresolved: list[str] = []
    for stem in targets:
        src = resolve_source(stem, by_stem)
        if src is None:
            unresolved.append(stem)
            continue
        fps_name = fps_from_filename(src.name)
        if fps_name is not None:
            lookup[stem] = {
                "fps": fps_name,
                "source_file": str(src.relative_to(videos_root)),
                "source_kind": "filename",
            }
            continue
        fps_p = fps_from_probe(src)
        if fps_p is None:
            lookup[stem] = {
                "fps": None,
                "source_file": str(src.relative_to(videos_root)),
                "source_kind": "probe-failed",
            }
        else:
            lookup[stem] = {
                "fps": round(fps_p, 6),
                "source_file": str(src.relative_to(videos_root)),
                "source_kind": "probe",
            }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(lookup, indent=2), encoding="utf-8")

    by_kind = {"filename": 0, "probe": 0, "probe-failed": 0}
    for v in lookup.values():
        by_kind[v["source_kind"]] += 1
    fps_hist: dict[str, int] = {}
    for v in lookup.values():
        f = v["fps"]
        if f is None:
            continue
        bucket = str(int(round(f)))
        fps_hist[bucket] = fps_hist.get(bucket, 0) + 1

    return {
        "n_total": len(targets),
        "n_resolved": len(lookup),
        "n_unresolved": len(unresolved),
        "by_kind": by_kind,
        "fps_hist": dict(sorted(fps_hist.items(), key=lambda kv: int(kv[0]))),
        "unresolved": unresolved,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--videos-root", type=Path,
        default=Path("../affwild2/videos"),
        help="Root containing batch1/, batch2/, new_vids/ subdirs.",
    )
    ap.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Optional visual-feature cache dir; restricts the lookup "
             "to the 307 MTL-release videos. Omit to probe every "
             "source video under videos-root.",
    )
    ap.add_argument(
        "--out", type=Path,
        default=Path("data/affwild2/fps_lookup.json"),
    )
    args = ap.parse_args()
    summary = build(args.videos_root, args.cache_dir, args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
