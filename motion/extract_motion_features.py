#!/usr/bin/env python3
"""Encode a deterministic Charades subset to H.264 and plot motion features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_VIDEO_DIR = Path("/workspace/s/lzw/datasets/TimeLens-Bench/videos/charades")
DEFAULT_ANNOTATION = Path("/workspace/s/lzw/datasets/TimeLens-Bench/charades-timelens.json")
DEFAULT_OUTPUT_DIR = Path("/workspace/s/lzw/datasets/TimeLens-Bench-H264/charades")
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}
ATOMIC_STATE_COLORS = {
    "RISING": "#a1d99b",
    "FALLING": "#fcae91",
    "STABLE": "#d9d9d9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically select a percentage of annotated videos, encode them "
            "to H.264, extract codec motion vectors, aggregate them to 1 fps, and "
            "draw the annotated event boundaries."
        )
    )
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="PNG output directory; defaults to --output-dir.",
    )
    parser.add_argument(
        "--percentage",
        type=float,
        default=20.0,
        help="Percentage of eligible videos to process (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used by deterministic subset selection (default: 42).",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=0,
        help="Threads per ffmpeg process; 0 lets ffmpeg choose.",
    )
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument(
        "--change-threshold",
        type=float,
        default=0.1,
        help="Absolute per-second motion change threshold (default: 0.1).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace existing encoded videos and plots.",
    )
    parser.add_argument(
        "--overwrite-plots",
        action="store_true",
        help="Redraw existing plots while reusing encoded videos.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.percentage <= 100.0:
        raise ValueError("--percentage must be in (0, 100].")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.ffmpeg_threads < 0:
        raise ValueError("--ffmpeg-threads cannot be negative.")
    if not 0 <= args.crf <= 51:
        raise ValueError("--crf must be between 0 and 51.")
    if args.change_threshold < 0.0:
        raise ValueError("--change-threshold cannot be negative.")
    if not args.video_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {args.video_dir}")
    if not args.annotation.is_file():
        raise FileNotFoundError(f"Annotation file not found: {args.annotation}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not available on PATH.")


def load_annotations(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("The annotation root must be a JSON object keyed by video ID.")
    return data


def discover_videos(video_dir: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    grouped: dict[str, list[Path]] = {}
    for path in sorted(video_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            grouped.setdefault(path.stem, []).append(path)
    unique = {video_id: paths[0] for video_id, paths in grouped.items() if len(paths) == 1}
    conflicts = {video_id: paths for video_id, paths in grouped.items() if len(paths) > 1}
    return unique, conflicts


def select_subset(
    videos: dict[str, Path],
    annotations: dict[str, dict[str, Any]],
    percentage: float,
    seed: int,
) -> list[tuple[str, Path, dict[str, Any]]]:
    eligible = sorted(set(videos).intersection(annotations))
    if not eligible:
        return []
    count = max(1, math.ceil(len(eligible) * percentage / 100.0))

    def rank(video_id: str) -> bytes:
        return hashlib.sha256(f"{seed}:{video_id}".encode("utf-8")).digest()

    selected_ids = sorted(sorted(eligible, key=rank)[:count])
    return [(video_id, videos[video_id], annotations[video_id]) for video_id in selected_ids]


def encode_h264(
    source: Path,
    destination: Path,
    preset: str,
    crf: int,
    ffmpeg_threads: int,
    overwrite: bool,
) -> str:
    if destination.exists() and not overwrite:
        return "reused"

    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}.", suffix=".mp4", dir=destination.parent, delete=False
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
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-bf",
        "0",
        "-refs",
        "1",
        "-g",
        "10000",
        "-keyint_min",
        "10000",
        "-sc_threshold",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if ffmpeg_threads > 0:
        command.extend(["-threads", str(ffmpeg_threads)])
    command.append(str(temporary))

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        os.replace(temporary, destination)
    except subprocess.CalledProcessError as error:
        message = error.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed for {source.name}: {message}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return "encoded"


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    threshold = quantile * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, threshold, side="left")), len(values) - 1)
    return float(sorted_values[index])


def motion_vector_feature(motion_vectors: Iterable[Any], delta_time: float) -> float | None:
    dx_values: list[float] = []
    dy_values: list[float] = []
    areas: list[float] = []
    for vector in motion_vectors:
        scale = float(vector.motion_scale)
        area = float(vector.w) * float(vector.h)
        if scale <= 0.0 or area <= 0.0:
            continue
        dx_values.append(float(vector.motion_x) / scale / delta_time)
        dy_values.append(float(vector.motion_y) / scale / delta_time)
        areas.append(area)

    if not areas:
        return None

    dx = np.asarray(dx_values, dtype=np.float64)
    dy = np.asarray(dy_values, dtype=np.float64)
    weights = np.asarray(areas, dtype=np.float64)
    global_dx = weighted_quantile(dx, weights, 0.5)
    global_dy = weighted_quantile(dy, weights, 0.5)
    residual = np.hypot(dx - global_dx, dy - global_dy)
    cap = weighted_quantile(residual, weights, 0.95)
    clipped = np.minimum(residual, cap)
    energy = math.sqrt(float(np.sum(weights * clipped**2) / np.sum(weights)))
    return math.log1p(energy)


def enable_motion_vector_export(codec_context: Any) -> None:
    if hasattr(codec_context, "export_mvs"):
        codec_context.export_mvs = True
        return
    options = dict(codec_context.options or {})
    options["flags2"] = "+export_mvs"
    codec_context.options = options


def extract_frame_features(video_path: Path) -> list[tuple[float, float]]:
    try:
        import av
    except ImportError as error:
        raise RuntimeError("PyAV is required; install motion/requirements.txt.") from error

    features: list[tuple[float, float]] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        enable_motion_vector_export(stream.codec_context)
        fallback_delta = 1.0 / float(stream.average_rate) if stream.average_rate else 1.0 / 25.0
        first_timestamp: float | None = None
        previous_timestamp: float | None = None

        for frame in container.decode(stream):
            if frame.time is not None:
                absolute_timestamp = float(frame.time)
            elif frame.pts is not None and frame.time_base is not None:
                absolute_timestamp = float(frame.pts * frame.time_base)
            else:
                absolute_timestamp = (
                    0.0 if previous_timestamp is None else previous_timestamp + fallback_delta
                )
            if first_timestamp is None:
                first_timestamp = absolute_timestamp
            timestamp = max(0.0, absolute_timestamp - first_timestamp)
            delta_time = fallback_delta
            if previous_timestamp is not None and absolute_timestamp > previous_timestamp:
                delta_time = absolute_timestamp - previous_timestamp
            previous_timestamp = absolute_timestamp

            motion_vectors = frame.side_data.get("MOTION_VECTORS")
            if motion_vectors is None:
                continue
            feature = motion_vector_feature(motion_vectors, delta_time)
            if feature is not None and math.isfinite(feature):
                features.append((timestamp, feature))

    if not features:
        raise RuntimeError(
            f"No motion-vector side data was decoded from {video_path.name}; "
            "check the PyAV/FFmpeg build and H.264 decoder."
        )
    return features


def aggregate_to_one_fps(
    frame_features: Sequence[tuple[float, float]], duration: float
) -> tuple[np.ndarray, np.ndarray]:
    bin_count = max(1, math.ceil(duration))
    buckets: list[list[float]] = [[] for _ in range(bin_count)]
    for timestamp, value in frame_features:
        index = int(math.floor(timestamp))
        if 0 <= index < bin_count:
            buckets[index].append(value)

    values = np.full(bin_count, np.nan, dtype=np.float64)
    for index, bucket in enumerate(buckets):
        if bucket:
            values[index] = float(np.percentile(bucket, 90))

    valid = np.flatnonzero(np.isfinite(values))
    if valid.size == 0:
        raise RuntimeError("No valid motion features fall inside the annotated duration.")
    missing = np.flatnonzero(~np.isfinite(values))
    if missing.size:
        values[missing] = np.interp(missing, valid, values[valid])
    timestamps = np.arange(bin_count, dtype=np.float64)
    return timestamps, values


def split_atomic_states_by_threshold(
    timestamps: np.ndarray,
    values: np.ndarray,
    duration: float,
    threshold: float,
) -> list[tuple[float, float, str]]:
    if len(values) <= 1:
        return [(0.0, float(duration), "STABLE")]

    differences = np.diff(values)
    edge_states = np.full(len(differences), "STABLE", dtype=object)
    edge_states[differences > threshold] = "RISING"
    edge_states[differences < -threshold] = "FALLING"

    stable_start = 0
    while stable_start < len(edge_states):
        if edge_states[stable_start] != "STABLE":
            stable_start += 1
            continue
        stable_end = stable_start + 1
        while stable_end < len(edge_states) and edge_states[stable_end] == "STABLE":
            stable_end += 1

        left_state = str(edge_states[stable_start - 1]) if stable_start > 0 else None
        right_state = str(edge_states[stable_end]) if stable_end < len(edge_states) else None
        replacement: str | None = None
        if left_state == right_state and left_state in {"RISING", "FALLING"}:
            replacement = left_state
        elif left_state is None and right_state in {"RISING", "FALLING"}:
            replacement = right_state
        elif right_state is None and left_state in {"RISING", "FALLING"}:
            replacement = left_state
        if replacement is not None:
            edge_states[stable_start:stable_end] = replacement
        stable_start = stable_end

    segments: list[tuple[float, float, str]] = []
    run_start = 0
    current_state = str(edge_states[0])
    for edge_index in range(1, len(edge_states)):
        state = str(edge_states[edge_index])
        if state == current_state:
            continue
        segments.append(
            (float(timestamps[run_start]), float(timestamps[edge_index]), current_state)
        )
        run_start = edge_index
        current_state = state
    segments.append((float(timestamps[run_start]), float(duration), current_state))
    return segments


def validate_annotation(video_id: str, annotation: dict[str, Any]) -> tuple[float, list, list]:
    duration = float(annotation["duration"])
    spans = annotation.get("spans", [])
    queries = annotation.get("queries", [])
    if duration <= 0.0:
        raise ValueError(f"{video_id}: duration must be positive.")
    if len(spans) != len(queries):
        raise ValueError(f"{video_id}: spans and queries must have the same length.")
    for span in spans:
        if not isinstance(span, list) or len(span) != 2:
            raise ValueError(f"{video_id}: every span must be [start, end].")
    return duration, spans, queries


def plot_motion_distribution(
    video_id: str,
    timestamps: np.ndarray,
    values: np.ndarray,
    duration: float,
    spans: Sequence[Sequence[float]],
    queries: Sequence[str],
    atomic_segments: Sequence[tuple[float, float, str]],
    destination: Path,
    overwrite: bool,
) -> str:
    if destination.exists() and not overwrite:
        return "reused"

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(14, 6))
    axis.plot(
        timestamps,
        values,
        color="#1f77b4",
        linewidth=1.8,
        label="Motion feature",
        zorder=3,
    )

    shown_states: set[str] = set()
    for segment_start, segment_end, state in atomic_segments:
        shading_label = state if state not in shown_states else None
        shown_states.add(state)
        axis.axvspan(
            segment_start,
            segment_end,
            color=ATOMIC_STATE_COLORS[state],
            alpha=0.30,
            label=shading_label,
            zorder=0,
        )
        segment_width = segment_end - segment_start
        axis.text(
            (segment_start + segment_end) / 2.0,
            0.98,
            state,
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7,
            rotation=90 if segment_width < 4.0 else 0,
            color="#303030",
            zorder=5,
        )

    colors = plt.get_cmap("tab10")
    for index, (span, query) in enumerate(zip(spans, queries)):
        start = min(max(float(span[0]), 0.0), duration)
        end = min(max(float(span[1]), 0.0), duration)
        color = colors(index % 10)
        label_query = textwrap.shorten(str(query), width=72, placeholder="...")
        label = f"Event {index + 1} [{start:g}, {end:g}]s: {label_query}"
        axis.axvline(
            start,
            color=color,
            linestyle="--",
            linewidth=1.4,
            label=label,
            zorder=4,
        )
        axis.axvline(end, color=color, linestyle="--", linewidth=1.4, zorder=4)

    axis.set_xlim(0.0, duration)
    axis.set_xlabel("Timestamp (s)")
    axis.set_ylabel("Residual motion feature")
    axis.set_title(f"Threshold-based atomic motion states: {video_id}")
    axis.grid(axis="y", linestyle=":", alpha=0.35)
    if spans or atomic_segments:
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8)
        figure.tight_layout(rect=(0.0, 0.18, 1.0, 1.0))
    else:
        figure.tight_layout()

    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}.", suffix=".png", dir=destination.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        figure.savefig(temporary, format="png", dpi=160, bbox_inches="tight")
        os.replace(temporary, destination)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)
    return "plotted"


def process_video(task: dict[str, Any]) -> dict[str, str]:
    video_id = task["video_id"]
    source = Path(task["source"])
    encoded = Path(task["encoded"])
    plot = Path(task["plot"])
    duration, spans, queries = validate_annotation(video_id, task["annotation"])

    encode_status = encode_h264(
        source=source,
        destination=encoded,
        preset=task["preset"],
        crf=task["crf"],
        ffmpeg_threads=task["ffmpeg_threads"],
        overwrite=task["overwrite"],
    )
    overwrite_plot = task["overwrite"] or task["overwrite_plots"]
    if plot.exists() and not overwrite_plot:
        return {"video_id": video_id, "encode": encode_status, "plot": "reused"}

    frame_features = extract_frame_features(encoded)
    timestamps, values = aggregate_to_one_fps(frame_features, duration)
    atomic_segments = split_atomic_states_by_threshold(
        timestamps=timestamps,
        values=values,
        duration=duration,
        threshold=task["change_threshold"],
    )
    plot_status = plot_motion_distribution(
        video_id=video_id,
        timestamps=timestamps,
        values=values,
        duration=duration,
        spans=spans,
        queries=queries,
        atomic_segments=atomic_segments,
        destination=plot,
        overwrite=overwrite_plot,
    )
    state_counts: dict[str, int] = {}
    for _, _, state in atomic_segments:
        state_counts[state] = state_counts.get(state, 0) + 1
    state_summary = ",".join(
        f"{state}:{count}" for state, count in sorted(state_counts.items())
    )
    return {
        "video_id": video_id,
        "encode": encode_status,
        "plot": plot_status,
        "splits": str(max(0, len(atomic_segments) - 1)),
        "states": state_summary,
    }


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        annotations = load_annotations(args.annotation)
        videos, conflicts = discover_videos(args.video_dir)
        selected = select_subset(videos, annotations, args.percentage, args.seed)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    plot_dir = args.plot_dir or args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    eligible_count = len(set(videos).intersection(annotations))
    print(
        f"Discovered {len(videos)} unique videos; {eligible_count} have annotations; "
        f"selected {len(selected)} ({args.percentage:g}%, seed={args.seed})."
    )
    if conflicts:
        print(f"Skipped {len(conflicts)} video IDs with duplicate filenames.", file=sys.stderr)
    if not selected:
        print("Error: no eligible videos were selected.", file=sys.stderr)
        return 2

    tasks = [
        {
            "video_id": video_id,
            "source": str(source),
            "annotation": annotation,
            "encoded": str(args.output_dir / f"{video_id}.mp4"),
            "plot": str(plot_dir / f"{video_id}.motion.png"),
            "preset": args.preset,
            "crf": args.crf,
            "ffmpeg_threads": args.ffmpeg_threads,
            "change_threshold": args.change_threshold,
            "overwrite": args.overwrite,
            "overwrite_plots": args.overwrite_plots,
        }
        for video_id, source, annotation in selected
    ]

    failures = 0
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_video, task): task["video_id"] for task in tasks}
        for future in as_completed(futures):
            video_id = futures[future]
            try:
                result = future.result()
                completed += 1
                print(
                    f"[{completed}/{len(tasks)}] {video_id}: "
                    f"video={result['encode']}, plot={result['plot']}, "
                    f"atomic_splits={result.get('splits', 'reused')}, "
                    f"atomic_states={result.get('states', 'reused')}"
                )
            except Exception as error:
                failures += 1
                print(f"[{video_id}] failed: {error}", file=sys.stderr)

    print(f"Finished: {len(tasks) - failures} succeeded, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
