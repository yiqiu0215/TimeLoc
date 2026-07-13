import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from training.modeling.special_tokens import TIME_BIN_COUNT, format_time_token


DEFAULT_COVERAGE_THRESHOLD = 0.5
DEFAULT_EPS = 1e-6


TIME_REFINE_PROMPT_PREFIX = (
    "You are given a video with multiple visual temporal blocks.\n"
    "The special token after each visual temporal block indicates its\n"
    "normalized sampling timestamp, ranging from <time_000> to <time_300>.\n\n"
    "Event query: {query}\n"
)

TIME_REFINE_PROMPT_SUFFIX = (
    "\nSelect all visual temporal blocks matching the event query:\n"
    "'{query}'\n\n"
    "Return only their timestamp tokens in chronological order, without separators.\n"
    "Answer format:\n"
    "<vtg><fg><time_029><time_038><fg></vtg>\n"
    "If none:\n"
    "<vtg><fg><fg></vtg>\n"
)


@dataclass(frozen=True)
class FrameLabelResult:
    labels: torch.Tensor
    support_left: torch.Tensor
    support_right: torch.Tensor
    overlap: torch.Tensor
    coverage: torch.Tensor


def build_time_refine_prompt_parts(query: str) -> tuple[str, str]:
    query = str(query).strip()
    if not query:
        raise ValueError("TimeRefine query must be non-empty.")
    return (
        TIME_REFINE_PROMPT_PREFIX.format(query=query),
        TIME_REFINE_PROMPT_SUFFIX.format(query=query),
    )


def build_time_refine_user_content(query: str, video_content: dict) -> list[dict]:
    prefix, suffix = build_time_refine_prompt_parts(query)
    if not isinstance(video_content, dict) or video_content.get("type") != "video":
        raise ValueError("video_content must be a single video content dictionary.")
    return [
        {"type": "text", "text": prefix},
        video_content,
        {"type": "text", "text": suffix},
    ]


def normalize_single_span(span: Sequence[float] | Iterable[Sequence[float]]) -> tuple[float, float]:
    if isinstance(span, tuple):
        span = list(span)
    if isinstance(span, list) and len(span) == 2 and all(
        isinstance(value, (int, float)) for value in span
    ):
        spans = [span]
    elif isinstance(span, list) and len(span) == 1 and isinstance(span[0], (list, tuple)):
        spans = [list(span[0])]
    else:
        raise ValueError(
            "TimeRefine SFT requires exactly one span in [start, end] format, "
            f"got {span!r}."
        )
    if len(spans[0]) != 2:
        raise ValueError(f"Invalid span: {span!r}")
    start, end = float(spans[0][0]), float(spans[0][1])
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError(f"Span must contain finite values, got {span!r}.")
    if start < 0 or start > end:
        raise ValueError(f"Expected 0 <= start <= end, got {span!r}.")
    return start, end


def extract_temporal_block_timestamps(
    videos,
    temporal_patch_size: int = 2,
) -> torch.Tensor:
    """Extract one real timestamp for every Qwen2.5 temporal patch block."""

    if videos is None or len(videos) != 1:
        raise ValueError(
            "TimeRefine requires exactly one video input with metadata; "
            f"got {0 if videos is None else len(videos)}."
        )
    entry = videos[0]
    if not isinstance(entry, (list, tuple)) or len(entry) != 2:
        raise ValueError(
            "Qwen2.5-TimeLens video metadata must be a (video_tensor, metadata) tuple."
        )
    metadata = entry[1]
    if not isinstance(metadata, dict):
        raise ValueError("Video metadata must be a dictionary.")
    fps = float(metadata.get("fps", 0.0))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Video metadata must contain positive fps, got {fps}.")
    frame_indices = metadata.get("frames_indices")
    if frame_indices is None:
        raise ValueError("Video metadata is missing frames_indices.")
    if hasattr(frame_indices, "tolist"):
        frame_indices = frame_indices.tolist()
    frame_indices = list(frame_indices)
    temporal_patch_size = int(temporal_patch_size)
    if temporal_patch_size <= 0:
        raise ValueError(f"temporal_patch_size must be positive, got {temporal_patch_size}.")
    if not frame_indices or len(frame_indices) % temporal_patch_size != 0:
        raise ValueError(
            "The sampled frame count must be divisible by temporal_patch_size: "
            f"frames={len(frame_indices)}, temporal_patch_size={temporal_patch_size}."
        )

    timestamps = torch.tensor(
        [float(index) / fps for index in frame_indices[::temporal_patch_size]],
        dtype=torch.float32,
    )
    if timestamps.numel() == 0 or not torch.isfinite(timestamps).all():
        raise ValueError("Temporal block timestamps must be finite and non-empty.")
    if timestamps.numel() > 1 and not torch.all(timestamps[1:] > timestamps[:-1]):
        raise ValueError("Temporal block timestamps must be strictly increasing.")
    return timestamps


def quantize_time_bins(
    timestamps: torch.Tensor | Sequence[float],
    duration: float,
    time_bin_count: int = TIME_BIN_COUNT,
) -> torch.Tensor:
    if int(time_bin_count) != TIME_BIN_COUNT:
        raise ValueError(
            f"TimeRefine requires exactly {TIME_BIN_COUNT} time bins, got {time_bin_count}."
        )
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"duration must be positive and finite, got {duration}.")
    values = torch.as_tensor(timestamps, dtype=torch.float32)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("timestamps must be a non-empty one-dimensional tensor.")
    if not torch.isfinite(values).all() or torch.any(values < 0) or torch.any(values > duration):
        raise ValueError("timestamps must lie in [0, duration].")
    bins = torch.tensor(
        [max(0, min(TIME_BIN_COUNT - 1, int(round(float(value))))) for value in (300 * values / duration)],
        dtype=torch.long,
    )
    if bins.numel() > 1 and torch.any(bins[1:] < bins[:-1]):
        raise ValueError("Quantized time bins must be non-decreasing.")
    return bins


def compute_frame_support_intervals(
    timestamps: torch.Tensor | Sequence[float],
    duration: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"duration must be positive and finite, got {duration}.")
    values = torch.as_tensor(timestamps, dtype=torch.float32)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("timestamps must be a non-empty one-dimensional tensor.")
    if not torch.isfinite(values).all() or torch.any(values < 0) or torch.any(values > duration):
        raise ValueError("timestamps must lie in [0, duration].")
    if values.numel() > 1 and not torch.all(values[1:] > values[:-1]):
        raise ValueError("timestamps must be strictly increasing.")

    support_left = torch.empty_like(values)
    support_right = torch.empty_like(values)
    support_left[0] = 0.0
    support_right[-1] = duration
    if values.numel() > 1:
        midpoints = 0.5 * (values[:-1] + values[1:])
        support_left[1:] = midpoints
        support_right[:-1] = midpoints
    return support_left, support_right


def build_frame_labels(
    timestamps: torch.Tensor | Sequence[float],
    duration: float,
    gt_start: float,
    gt_end: float,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    eps: float = DEFAULT_EPS,
) -> FrameLabelResult:
    if not 0.0 <= float(coverage_threshold) <= 1.0:
        raise ValueError(
            f"coverage_threshold must be in [0, 1], got {coverage_threshold}."
        )
    gt_start, gt_end = float(gt_start), float(gt_end)
    duration = float(duration)
    if not 0 <= gt_start <= gt_end <= duration:
        raise ValueError(
            f"Expected 0 <= gt_start <= gt_end <= duration, got "
            f"[{gt_start}, {gt_end}], duration={duration}."
        )
    support_left, support_right = compute_frame_support_intervals(
        timestamps, duration
    )
    gt_start_tensor = torch.tensor(
        gt_start, dtype=support_left.dtype, device=support_left.device
    )
    gt_end_tensor = torch.tensor(
        gt_end, dtype=support_right.dtype, device=support_right.device
    )
    overlap = torch.clamp(
        torch.minimum(support_right, gt_end_tensor)
        - torch.maximum(support_left, gt_start_tensor),
        min=0.0,
    )
    coverage = overlap / (support_right - support_left + float(eps))
    labels = (coverage >= float(coverage_threshold)).to(torch.long)
    if int(labels.sum().item()) == 0:
        labels[int(torch.argmax(overlap).item())] = 1
    return FrameLabelResult(
        labels=labels,
        support_left=support_left,
        support_right=support_right,
        overlap=overlap,
        coverage=coverage,
    )


def build_vtg_target(frame_labels: torch.Tensor, frame_bin_ids: torch.Tensor) -> str:
    labels = torch.as_tensor(frame_labels, dtype=torch.long).reshape(-1)
    bins = torch.as_tensor(frame_bin_ids, dtype=torch.long).reshape(-1)
    if labels.numel() == 0 or labels.numel() != bins.numel():
        raise ValueError(
            "frame_labels and frame_bin_ids must be non-empty and have equal length."
        )
    if torch.any((labels != 0) & (labels != 1)):
        raise ValueError("frame_labels must contain only 0/1 values.")
    if torch.any(bins < 0) or torch.any(bins >= TIME_BIN_COUNT):
        raise ValueError("frame_bin_ids must lie in [0, 300].")
    tokens = ["<vtg>", "<fg>"]
    for label, time_bin in zip(labels.tolist(), bins.tolist()):
        if int(label) == 1:
            tokens.append(format_time_token(int(time_bin)))
    tokens.extend(("<fg>", "</vtg>"))
    return "".join(tokens)
