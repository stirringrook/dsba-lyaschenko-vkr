"""Download CREMA-D and RAVDESS into the interim-pipeline layout.

RAVDESS lives on Zenodo (record 1188976) as 24 per-actor zips. CREMA-D
lives in the ``CheyneyComputerScience/CREMA-D`` GitHub repository with
Git LFS for the media files.

Both sources are public and require no authentication. Total on-disk
footprint after unpacking is roughly 20 GB; see the individual function
docstrings for a per-source breakdown.

Nothing in this module should be imported at unit-test time: the functions
make network calls and shell out to ``git``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional


# ----------------------------------------------------------------------
# RAVDESS (Zenodo 1188976)
# ----------------------------------------------------------------------

RAVDESS_ZENODO_RECORD = "1188976"
RAVDESS_VIDEO_SPEECH_URL = (
    "https://zenodo.org/record/{rec}/files/Video_Speech_Actor_{actor:02d}.zip"
    "?download=1"
)


def _download_with_progress(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` with a best-effort progress line."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0)) or None
        downloaded = 0
        chunk = 1 << 20  # 1 MiB
        with tmp.open("wb") as fh:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                fh.write(buf)
                downloaded += len(buf)
                if total:
                    pct = 100.0 * downloaded / total
                    print(
                        f"  {dest.name}: {downloaded / 1e6:7.1f} MB / "
                        f"{total / 1e6:7.1f} MB  ({pct:5.1f}%)",
                        end="\r",
                        flush=True,
                    )
    tmp.replace(dest)
    print()  # newline after progress carriage-return


def download_ravdess_video_speech(
    output_dir: str | Path,
    actors: Optional[Iterable[int]] = None,
    keep_zips: bool = False,
) -> List[Path]:
    """Download RAVDESS Video_Speech zips and extract them.

    After the call, ``<output_dir>/Actor_XX/<clip>.mp4`` holds the speech
    clips for every requested actor. This is the layout
    :func:`src.datasets.interim_prep.parse_ravdess_clips` expects when
    called with ``video_dir=<output_dir>``.

    Args:
        output_dir: Destination; the per-actor subfolders land here.
        actors: Iterable of 1-24 integers. ``None`` means all 24.
        keep_zips: If ``True``, preserve the downloaded zips under
            ``<output_dir>/_zips/``. Default deletes them after unpack.

    Returns:
        List of per-actor directories created.
    """
    output_dir = Path(output_dir)
    zips_dir = output_dir / "_zips"
    output_dir.mkdir(parents=True, exist_ok=True)
    zips_dir.mkdir(parents=True, exist_ok=True)

    wanted = list(actors) if actors is not None else list(range(1, 25))
    created: List[Path] = []

    for a in wanted:
        actor_sub = output_dir / f"Actor_{a:02d}"
        if actor_sub.is_dir() and any(actor_sub.glob("*.mp4")):
            print(f"[ravdess] Actor_{a:02d}: already extracted, skipping.")
            created.append(actor_sub)
            continue
        url = RAVDESS_VIDEO_SPEECH_URL.format(rec=RAVDESS_ZENODO_RECORD, actor=a)
        zip_path = zips_dir / f"Video_Speech_Actor_{a:02d}.zip"
        if not zip_path.exists():
            print(f"[ravdess] downloading Actor_{a:02d}...")
            _download_with_progress(url, zip_path)
        print(f"[ravdess] extracting Actor_{a:02d}...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(output_dir)
        if not keep_zips:
            zip_path.unlink(missing_ok=True)
        created.append(actor_sub)

    if not keep_zips:
        # _zips/ is empty now; clean it up.
        try:
            zips_dir.rmdir()
        except OSError:
            pass

    return created


# ----------------------------------------------------------------------
# CREMA-D (GitHub + Git LFS)
# ----------------------------------------------------------------------

# The CheyneyComputerScience GitHub repo has permanently exhausted its
# LFS egress budget ("This repository exceeded its LFS budget" on every
# batch response). The README explicitly directs users to the maintainers'
# GitLab mirror when that happens, so we try that first.
CREMA_D_REPO_GITLAB = "https://gitlab.com/cs-cooper-lab/crema-d-mirror.git"
CREMA_D_REPO_GITHUB = "https://github.com/CheyneyComputerScience/CREMA-D.git"
CREMA_D_REPO = CREMA_D_REPO_GITLAB  # primary; GitHub is a secondary fallback.


def _run(
    cmd: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> None:
    """Run a subprocess, echoing the command and streaming its output.

    ``env`` entries are overlaid on top of ``os.environ`` so we can export
    (for example) ``GIT_LFS_SKIP_SMUDGE`` without clobbering PATH.
    """
    print(f"[cmd] {' '.join(cmd)} (cwd={cwd})")
    merged_env = None
    if env:
        merged_env = dict(os.environ)
        merged_env.update(env)
    subprocess.check_call(
        cmd, cwd=str(cwd) if cwd else None, env=merged_env,
    )


def _looks_like_lfs_pointer(path: Path) -> bool:
    """Return True if ``path`` is small and starts with the LFS v1 banner."""
    try:
        if path.stat().st_size > 1024:
            return False
        with path.open("rb") as fh:
            head = fh.read(64)
        return head.startswith(b"version https://git-lfs")
    except OSError:
        return False


def _assert_media_hydrated(root: Path, sub: str, ext: str) -> None:
    """Raise if the files under ``root/sub`` are still LFS pointer stubs.

    The GitHub "Download ZIP" button for an LFS repo returns pointer
    stubs, and a misconfigured ``git lfs pull`` can do the same. This
    check catches both before the caller hits an unreadable-media error
    several notebooks downstream.
    """
    d = root / sub
    if not d.is_dir():
        return
    sample = next(iter(d.glob(f"*{ext}")), None)
    if sample is not None and _looks_like_lfs_pointer(sample):
        raise RuntimeError(
            f"{d}/ contains Git LFS pointer stubs (e.g. {sample.name} is "
            f"{sample.stat().st_size} B), not real media. The LFS pull "
            f"did not hydrate the content. To diagnose, run:\n"
            f"    cd {d.parent}/_repo && git lfs pull\n"
            f"If the output contains 'exceeded its LFS budget', both the "
            f"GitLab mirror and the GitHub repo have run out of free egress "
            f"quota and you will need a third-party mirror (e.g. via the "
            f"CREMA-D maintainers at dcooper@wcupa.edu)."
        )


def download_crema_d(
    output_dir: str | Path,
    include_video_flash: bool = True,
    include_audio_wav: bool = True,
) -> Path:
    """Clone CREMA-D and place the media under ``output_dir``.

    The CheyneyComputerScience repository ships the media via Git LFS:
    without ``git lfs`` on PATH (or with its hooks uninitialised in the
    local repo), the files you get are ~130-byte pointer stubs, not real
    videos. This function:

    1. Checks for ``git`` and ``git-lfs`` on PATH.
    2. Clones with ``GIT_LFS_SKIP_SMUDGE=1`` so the initial clone is a
       tiny index. This is the official env-var escape hatch and avoids
       the ``-c filter.lfs.smudge=...`` trick, which is fragile on
       Windows where git dispatches the filter through a shell.
    3. Runs ``git lfs install --local`` inside the clone to guarantee
       the smudge/filter hooks are wired up before the pull.
    4. Pulls only the requested subdirectories via ``--include``.
    5. If that include-filtered pull fails (some git-lfs versions reject
       comma-separated patterns, and exit 2 is the usual symptom), falls
       back to a full ``git lfs pull``.
    6. Sanity-checks the result: if any media file is still a pointer
       stub, raises with an actionable message.

    Post-condition with the defaults:
        ``<output_dir>/VideoFlash/*.flv`` (7442 clips)
        ``<output_dir>/AudioWAV/*.wav``   (7442 clips)

    These are the exact directory names expected by
    :func:`src.datasets.interim_prep.parse_crema_d_clips` when called with
    ``video_dir=<output_dir>/VideoFlash``.

    Args:
        output_dir: Destination root (the function creates it).
        include_video_flash: If ``False``, skip fetching the ``.flv`` set.
        include_audio_wav: If ``False``, skip fetching the ``.wav`` set.

    Returns:
        ``output_dir`` as a ``Path``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("git") is None:
        raise FileNotFoundError("git not found on PATH; install Git first.")
    if shutil.which("git-lfs") is None:
        raise FileNotFoundError(
            "git-lfs not found on PATH. CREMA-D media is LFS-backed; "
            "install git-lfs (https://git-lfs.com) and re-run."
        )

    # Record version up-front so the cell output tells us *which* git-lfs
    # ran. The comma-separated --include syntax is very old, so this is
    # almost always fine, but having it in the log helps future-us.
    subprocess.check_call(["git-lfs", "--version"])

    repo_dir = output_dir / "_repo"
    if not repo_dir.is_dir():
        # Try the GitLab mirror first (GitHub's LFS budget for this repo
        # has been exhausted for years); fall back to GitHub only if the
        # GitLab mirror itself is unreachable.
        try:
            _run(
                ["git", "clone", "--depth", "1", CREMA_D_REPO_GITLAB, str(repo_dir)],
                env={"GIT_LFS_SKIP_SMUDGE": "1"},
            )
        except subprocess.CalledProcessError as e:
            print(f"[crema-d] GitLab mirror clone failed ({e}); trying GitHub.")
            _run(
                ["git", "clone", "--depth", "1", CREMA_D_REPO_GITHUB, str(repo_dir)],
                env={"GIT_LFS_SKIP_SMUDGE": "1"},
            )

    # Guarantee the local repo has LFS hooks even if the user's global
    # `git lfs install` was never run. Needed before `git lfs pull`.
    _run(["git", "lfs", "install", "--local"], cwd=repo_dir)

    lfs_includes: List[str] = []
    if include_video_flash:
        lfs_includes.append("VideoFlash")
    if include_audio_wav:
        lfs_includes.append("AudioWAV")

    if lfs_includes:
        try:
            _run(
                ["git", "lfs", "pull", "--include", ",".join(lfs_includes)],
                cwd=repo_dir,
            )
        except subprocess.CalledProcessError as e:
            print(
                f"[crema-d] include-filtered pull failed ({e}); "
                f"falling back to a full `git lfs pull` (~12 GB)."
            )
            _run(["git", "lfs", "pull"], cwd=repo_dir)

    # Move (not copy) the media directories to the target location so we
    # don't double disk usage.
    for sub in ("VideoFlash", "AudioWAV"):
        src = repo_dir / sub
        dst = output_dir / sub
        if src.is_dir() and not dst.exists():
            print(f"[crema-d] moving {sub}/ into place ...")
            src.rename(dst)

    # Catch the two common failure modes (LFS mis-fetch, ZIP-from-GitHub
    # pointer stubs) before the caller hits a cryptic decode error.
    if include_video_flash:
        _assert_media_hydrated(output_dir, "VideoFlash", ".flv")
    if include_audio_wav:
        _assert_media_hydrated(output_dir, "AudioWAV", ".wav")

    return output_dir


# ----------------------------------------------------------------------
# Thin CLI
# ----------------------------------------------------------------------

def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="Download CREMA-D and/or RAVDESS for the interim pipeline."
    )
    p.add_argument("--ravdess-dir", help="Destination for RAVDESS Video_Speech/")
    p.add_argument(
        "--ravdess-actors", default="",
        help="Comma-separated actor IDs (1-24). Empty = all 24."
    )
    p.add_argument("--crema-dir", help="Destination for CREMA-D root.")
    p.add_argument("--keep-zips", action="store_true", help="Keep RAVDESS zips.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.ravdess_dir:
        actors = (
            [int(s) for s in args.ravdess_actors.split(",") if s.strip()]
            if args.ravdess_actors else None
        )
        download_ravdess_video_speech(
            args.ravdess_dir, actors=actors, keep_zips=args.keep_zips,
        )
    if args.crema_dir:
        download_crema_d(args.crema_dir)


if __name__ == "__main__":
    main()
