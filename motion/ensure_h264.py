#!/usr/bin/env python3
"""Check videos in a directory and replace non-H.264 files with H.264 versions."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".avi",
    ".m2ts",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".mts",
    ".ts",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively check videos in a directory and atomically replace "
            "non-H.264 videos after transcoding them with libx264."
        )
    )
    parser.add_argument("video_dir", type=Path, help="Directory containing videos.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent ffmpeg jobs.")
    parser.add_argument("--ffmpeg-threads", type=int, default=0)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report which videos need transcoding; do not modify files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.video_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {args.video_dir}")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.ffmpeg_threads < 0:
        raise ValueError("--ffmpeg-threads cannot be negative.")
    if not 0 <= args.crf <= 51:
        raise ValueError("--crf must be between 0 and 51.")
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"{executable} is not available on PATH.")


def discover_videos(video_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(video_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def probe_video_codecs(path: Path) -> list[str]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    streams = json.loads(result.stdout).get("streams", [])
    codecs = [stream.get("codec_name", "unknown") for stream in streams]
    if not codecs:
        raise RuntimeError("no video stream found")
    return codecs


def transcode_and_replace(
    source: Path,
    preset: str,
    crf: int,
    ffmpeg_threads: int,
) -> None:
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{source.stem}.h264.",
        suffix=source.suffix,
        dir=source.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-map",
        "0",
        "-map_metadata",
        "0",
        "-map_chapters",
        "0",
        "-c",
        "copy",
        "-c:v:0",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-vf:v:0",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt:v:0",
        "yuv420p",
        "-threads",
        str(ffmpeg_threads),
        str(temporary),
    ]

    try:
        subprocess.run(command, check=True)
        codecs = probe_video_codecs(temporary)
        if any(codec != "h264" for codec in codecs):
            raise RuntimeError(f"output video codecs are not all H.264: {codecs}")
        os.replace(temporary, source)
    finally:
        temporary.unlink(missing_ok=True)


def process_video(path: Path, args: argparse.Namespace) -> tuple[str, str]:
    try:
        codecs = probe_video_codecs(path)
        if all(codec == "h264" for codec in codecs):
            return "h264", str(path)
        if args.dry_run:
            return "needs_conversion", f"{path} (codecs: {', '.join(codecs)})"
        transcode_and_replace(path, args.preset, args.crf, args.ffmpeg_threads)
        return "converted", str(path)
    except (
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as error:
        return "failed", f"{path}: {error}"


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}")
        return 2

    videos = discover_videos(args.video_dir)
    if not videos:
        print(f"No supported videos found in: {args.video_dir}")
        return 0

    counts = {"h264": 0, "needs_conversion": 0, "converted": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_video, path, args): path for path in videos}
        for future in as_completed(futures):
            status, message = future.result()
            counts[status] += 1
            print(f"[{status.upper()}] {message}", flush=True)

    print(
        "Summary: "
        f"total={len(videos)}, h264={counts['h264']}, "
        f"needs_conversion={counts['needs_conversion']}, "
        f"converted={counts['converted']}, failed={counts['failed']}"
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
