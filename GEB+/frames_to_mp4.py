#!/usr/bin/env python3
"""Batch-convert timestamp-named GEB+ frames into CFR H.264 MP4 videos."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


DEFAULT_FRAME_ROOT = Path("/workspace/s/lzw/datasets/GEB+/frames/frame")
DEFAULT_ANNOTATION = Path("/workspace/s/lzw/datasets/GEB+/train.json")
DEFAULT_OUTPUT_ROOT = Path("/workspace/s/lzw/datasets/GEB+/videos")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COMMON_FRAME_RATES = (
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    8.0,
    10.0,
    12.0,
    15.0,
    20.0,
    24000 / 1001,
    24.0,
    25.0,
    30000 / 1001,
    30.0,
)


@dataclass(frozen=True)
class Frame:
    timestamp: float
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert every video directory under the GEB+ frame root into one "
            "constant-frame-rate H.264 MP4. Image stems must be timestamps in seconds."
        )
    )
    parser.add_argument("--frame-root", type=Path, default=DEFAULT_FRAME_ROOT)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--fps",
        default="auto",
        help="Output FPS, or 'auto' to infer it separately for each video (default: auto).",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ffmpeg-threads", type=int, default=1)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace MP4 files that already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the planned conversions without running ffmpeg.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> float | None:
    if not args.frame_root.is_dir():
        raise FileNotFoundError(f"Frame root does not exist: {args.frame_root}")
    if args.annotation is not None and not args.annotation.is_file():
        raise FileNotFoundError(f"Annotation file does not exist: {args.annotation}")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.ffmpeg_threads < 1:
        raise ValueError("--ffmpeg-threads must be at least 1.")
    if not 0 <= args.crf <= 51:
        raise ValueError("--crf must be between 0 and 51.")
    if shutil.which("ffmpeg") is None and not args.dry_run:
        raise RuntimeError("ffmpeg is not available on PATH.")

    if args.fps.lower() == "auto":
        return None
    fps = float(args.fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("--fps must be 'auto' or a positive number.")
    return fps


def load_annotation_end_times(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as file:
        annotations = json.load(file)
    if not isinstance(annotations, dict):
        raise ValueError("The annotation root must be an object keyed by video ID.")

    end_times: dict[str, float] = {}
    for video_id, boundaries in annotations.items():
        if not isinstance(boundaries, list) or not boundaries:
            continue
        candidates = [
            float(boundary["next_timestamp"])
            for boundary in boundaries
            if boundary.get("next_timestamp") is not None
        ]
        if candidates:
            end_times[video_id] = max(candidates)
    return end_times


def discover_video_directories(frame_root: Path) -> list[Path]:
    return sorted(path for path in frame_root.iterdir() if path.is_dir())


def load_frames(video_dir: Path) -> list[Frame]:
    image_paths = sorted(
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError("no supported image files")

    frames: list[Frame] = []
    invalid_names: list[str] = []
    for path in image_paths:
        try:
            timestamp = float(path.stem)
        except ValueError:
            invalid_names.append(path.name)
            continue
        if not math.isfinite(timestamp) or timestamp < 0:
            invalid_names.append(path.name)
            continue
        frames.append(Frame(timestamp=timestamp, path=path.resolve()))

    if invalid_names:
        preview = ", ".join(invalid_names[:5])
        raise ValueError(f"invalid timestamp filenames: {preview}")
    if len(frames) < 2:
        raise ValueError("at least two timestamped frames are required")

    frames.sort(key=lambda frame: (frame.timestamp, frame.path.name))
    duplicates = [
        frames[index].timestamp
        for index in range(1, len(frames))
        if math.isclose(
            frames[index - 1].timestamp,
            frames[index].timestamp,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ]
    if duplicates:
        raise ValueError(f"duplicate timestamp: {duplicates[0]:.9f}")
    return frames


def infer_fps(frames: list[Frame]) -> float:
    deltas = [
        current.timestamp - previous.timestamp
        for previous, current in zip(frames, frames[1:])
        if current.timestamp - previous.timestamp > 1e-6
    ]
    if not deltas:
        raise ValueError("cannot infer FPS because timestamps do not increase")

    raw_fps = 1.0 / statistics.median(deltas)
    nearest = min(COMMON_FRAME_RATES, key=lambda candidate: abs(candidate - raw_fps))
    if abs(nearest - raw_fps) / nearest <= 0.02:
        return nearest
    return round(raw_fps, 6)


def ffconcat_escape(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "'\\''")


def write_concat_manifest(
    manifest_path: Path,
    frames: list[Frame],
    fps: float,
    annotated_end: float | None,
) -> float:
    nominal_frame_duration = 1.0 / fps
    final_end = frames[-1].timestamp + nominal_frame_duration
    if annotated_end is not None and annotated_end > frames[-1].timestamp:
        final_end = annotated_end

    entries: list[tuple[Path, float]] = []
    if frames[0].timestamp > 1e-6:
        entries.append((frames[0].path, frames[0].timestamp))
    for current, following in zip(frames, frames[1:]):
        entries.append((current.path, following.timestamp - current.timestamp))
    entries.append((frames[-1].path, final_end - frames[-1].timestamp))

    with manifest_path.open("w", encoding="utf-8", newline="\n") as file:
        file.write("ffconcat version 1.0\n")
        for frame_path, duration in entries:
            file.write(f"file '{ffconcat_escape(frame_path)}'\n")
            file.write(f"duration {duration:.9f}\n")
        # The concat demuxer ignores the final duration unless the last file is repeated.
        file.write(f"file '{ffconcat_escape(frames[-1].path)}'\n")
    return final_end


def convert_video(
    video_dir: Path,
    output_root: Path,
    requested_fps: float | None,
    annotated_end: float | None,
    preset: str,
    crf: int,
    ffmpeg_threads: int,
    overwrite: bool,
    dry_run: bool,
) -> tuple[str, str, int, float, float]:
    video_id = video_dir.name
    frames = load_frames(video_dir)
    fps = requested_fps if requested_fps is not None else infer_fps(frames)
    output_path = output_root / f"{video_id}.mp4"

    if output_path.exists() and not overwrite:
        return video_id, "skipped", len(frames), fps, 0.0
    if dry_run:
        duration = annotated_end or frames[-1].timestamp + 1.0 / fps
        return video_id, "planned", len(frames), fps, duration

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{video_id}.", dir=output_root) as temp_dir:
        temp_dir_path = Path(temp_dir)
        manifest_path = temp_dir_path / "frames.ffconcat"
        temporary_output = temp_dir_path / f"{video_id}.mp4"
        duration = write_concat_manifest(manifest_path, frames, fps, annotated_end)

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest_path),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            (
                f"fps=fps={fps:.9f}:start_time=0,"
                "scale=trunc(iw/2)*2:trunc(ih/2)*2"
            ),
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-threads",
            str(ffmpeg_threads),
            str(temporary_output),
        ]
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as error:
            message = error.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(message or "ffmpeg failed without an error message") from error
        os.replace(temporary_output, output_path)

    return video_id, "converted", len(frames), fps, duration


def main() -> int:
    args = parse_args()
    requested_fps = validate_args(args)
    annotation_end_times = load_annotation_end_times(args.annotation)
    video_dirs = discover_video_directories(args.frame_root)
    if not video_dirs:
        raise RuntimeError(f"No video directories found under {args.frame_root}")

    print(f"Frame root : {args.frame_root}")
    print(f"Output root: {args.output_root}")
    print(f"Videos     : {len(video_dirs)}")
    print(f"FPS        : {requested_fps if requested_fps is not None else 'auto'}")

    counts = {"converted": 0, "skipped": 0, "planned": 0, "failed": 0}
    futures = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for video_dir in video_dirs:
            future = executor.submit(
                convert_video,
                video_dir,
                args.output_root,
                requested_fps,
                annotation_end_times.get(video_dir.name),
                args.preset,
                args.crf,
                args.ffmpeg_threads,
                args.overwrite,
                args.dry_run,
            )
            futures[future] = video_dir.name

        for completed, future in enumerate(as_completed(futures), start=1):
            video_id = futures[future]
            try:
                _, status, frame_count, fps, duration = future.result()
                counts[status] += 1
                print(
                    f"[{completed}/{len(futures)}] {status:9s} {video_id} "
                    f"frames={frame_count} fps={fps:.6g} duration={duration:.3f}s"
                )
            except Exception as error:
                counts["failed"] += 1
                print(
                    f"[{completed}/{len(futures)}] failed    {video_id}: {error}",
                    flush=True,
                )

    print(
        "Summary: "
        + ", ".join(f"{key}={value}" for key, value in counts.items())
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
