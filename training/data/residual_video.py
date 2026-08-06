import math
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info, smart_resize
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional


def _metadata_value(metadata, name):
    if isinstance(metadata, dict):
        return metadata[name]
    return getattr(metadata, name)


def _find_video_content(messages):
    video_contents = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        video_contents.extend(
            item
            for item in content
            if isinstance(item, dict) and item.get("type") == "video"
        )
    if len(video_contents) != 1:
        raise ValueError(
            f"RIT preprocessing currently requires exactly one video, got {len(video_contents)}."
        )
    return video_contents[0]


def _decode_dense_frames(video_path: str, frame_indices: list[int]) -> torch.Tensor:
    if video_path.startswith("file://"):
        video_path = video_path[7:]
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Residual source video not found: {video_path}")

    import decord

    reader = decord.VideoReader(video_path)
    max_index = len(reader) - 1
    clipped_indices = [min(max(int(index), 0), max_index) for index in frame_indices]
    frames = reader.get_batch(clipped_indices).asnumpy()
    return torch.from_numpy(frames).permute(0, 3, 1, 2)


def _normalize_frames(frames: torch.Tensor, video_processor) -> torch.Tensor:
    frames = frames.to(torch.float32)
    if video_processor.do_rescale:
        frames = frames * float(getattr(video_processor, "rescale_factor", 1 / 255.0))
    if video_processor.do_normalize:
        mean = torch.tensor(
            video_processor.image_mean, dtype=frames.dtype, device=frames.device
        ).view(1, -1, 1, 1)
        std = torch.tensor(
            video_processor.image_std, dtype=frames.dtype, device=frames.device
        ).view(1, -1, 1, 1)
        frames = (frames - mean) / std
    return frames


def _pack_residual_patches(
    residuals: torch.Tensor, patch_size: int, merge_size: int
) -> torch.Tensor:
    num_blocks, channels, height, width = residuals.shape
    grid_h = height // patch_size
    grid_w = width // patch_size
    patches = residuals.reshape(
        num_blocks,
        channels,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
    return patches.reshape(
        num_blocks * grid_h * grid_w, channels * patch_size * patch_size
    )


def _format_timestamp(timestamp: float) -> str:
    return f"{timestamp:.6f}".rstrip("0").rstrip(".")


def _build_interleaved_placeholder(processor, midpoints, tokens_per_block):
    vision_start = getattr(processor, "vision_start_token", "<|vision_start|>")
    vision_end = getattr(processor, "vision_end_token", "<|vision_end|>")
    video_token = getattr(processor, "video_token", "<|video_pad|>")
    return "".join(
        f"<{_format_timestamp(float(midpoint))} seconds>"
        + vision_start
        + video_token * tokens_per_block
        + vision_end
        for midpoint in midpoints
    )


def prepare_rit_video_inputs(
    processor,
    messages,
    *,
    residual_num_diffs: int,
    min_tokens: int,
    total_tokens: int,
    add_generation_prompt: bool = False,
):
    if residual_num_diffs <= 0:
        raise ValueError("residual_num_diffs must be positive.")

    video_content = _find_video_content(messages)
    images, decoded_videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if images is not None:
        raise ValueError("RIT training currently supports video-only visual inputs.")
    if decoded_videos is None or len(decoded_videos) != 1:
        raise ValueError("RIT preprocessing expected one decoded video.")

    base_video, metadata = decoded_videos[0]
    frame_indices = _metadata_value(metadata, "frames_indices")
    if hasattr(frame_indices, "tolist"):
        frame_indices = frame_indices.tolist()
    frame_indices = [int(index) for index in frame_indices]
    if len(frame_indices) != base_video.shape[0]:
        raise ValueError("Decoded RGB frames and metadata frame indices do not match.")

    temporal_patch_size = int(processor.video_processor.temporal_patch_size)
    if temporal_patch_size != 2:
        raise ValueError(
            f"RIT currently requires temporal_patch_size=2, got {temporal_patch_size}."
        )
    num_rgb_blocks = math.ceil(base_video.shape[0] / temporal_patch_size)
    num_interleaved_blocks = 2 * num_rgb_blocks - 1
    if num_interleaved_blocks * min_tokens > total_tokens:
        raise ValueError(
            "Combined visual token budget is too small for the requested minimum: "
            f"{num_interleaved_blocks} * {min_tokens} > {total_tokens}."
        )

    patch_size = int(processor.video_processor.patch_size)
    merge_size = int(processor.video_processor.merge_size)
    spatial_factor = patch_size * merge_size
    max_pixels_per_block = (total_tokens // num_interleaved_blocks) * spatial_factor**2
    min_pixels_per_block = min_tokens * spatial_factor**2
    resized_height, resized_width = smart_resize(
        int(base_video.shape[-2]),
        int(base_video.shape[-1]),
        factor=spatial_factor,
        min_pixels=min_pixels_per_block,
        max_pixels=max_pixels_per_block,
    )
    base_video = vision_functional.resize(
        base_video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).to(torch.float32)

    rgb_inputs = processor.video_processor(
        videos=[base_video],
        video_metadata=[metadata],
        return_tensors="pt",
        return_metadata=True,
        do_resize=False,
        do_sample_frames=video_kwargs.get("do_sample_frames", False),
    )
    rgb_video_grid_thw = rgb_inputs["video_grid_thw"]
    rgb_grid_t, grid_h, grid_w = [
        int(value) for value in rgb_video_grid_thw[0].tolist()
    ]
    if rgb_grid_t != num_rgb_blocks:
        raise ValueError(
            f"RGB grid has {rgb_grid_t} blocks, expected {num_rgb_blocks}."
        )

    raw_fps = float(_metadata_value(metadata, "fps"))
    if not math.isfinite(raw_fps) or raw_fps <= 0:
        raise ValueError(f"Invalid source video FPS: {raw_fps}")
    base_timestamps = [index / raw_fps for index in frame_indices]
    padded_timestamps = list(base_timestamps)
    while len(padded_timestamps) < num_rgb_blocks * temporal_patch_size:
        padded_timestamps.append(padded_timestamps[-1])
    rgb_midpoints = [
        (padded_timestamps[2 * index] + padded_timestamps[2 * index + 1]) / 2
        for index in range(num_rgb_blocks)
    ]

    dense_indices = []
    for block_index in range(num_rgb_blocks - 1):
        start_index = frame_indices[2 * block_index + 1]
        end_index = frame_indices[2 * block_index + 2]
        dense_indices.extend(
            torch.linspace(
                start_index, end_index, residual_num_diffs + 1
            ).round().to(torch.long).tolist()
        )

    residual_midpoints = [
        (base_timestamps[2 * index + 1] + base_timestamps[2 * index + 2]) / 2
        for index in range(num_rgb_blocks - 1)
    ]
    if dense_indices:
        dense_frames = _decode_dense_frames(video_content["video"], dense_indices)
        dense_frames = vision_functional.resize(
            dense_frames,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        dense_frames = _normalize_frames(dense_frames, processor.video_processor)
        dense_frames = dense_frames.reshape(
            num_rgb_blocks - 1,
            residual_num_diffs + 1,
            3,
            resized_height,
            resized_width,
        )
        residual_steps = dense_frames[:, 1:] - dense_frames[:, :-1]
        residuals = residual_steps.sum(dim=1)
        pixel_values_residuals = _pack_residual_patches(
            residuals, patch_size=patch_size, merge_size=merge_size
        )
    else:
        pixel_values_residuals = torch.empty(
            (0, 3 * patch_size**2), dtype=torch.float32
        )

    residual_grid_thw = torch.tensor(
        [[num_rgb_blocks - 1, grid_h, grid_w]], dtype=torch.long
    )
    video_grid_thw = torch.tensor(
        [[num_interleaved_blocks, grid_h, grid_w]], dtype=torch.long
    )
    merged_visual_tokens = num_interleaved_blocks * grid_h * grid_w // merge_size**2
    if merged_visual_tokens > total_tokens:
        raise ValueError(
            f"Combined visual tokens exceed budget: {merged_visual_tokens} > {total_tokens}."
        )

    interleaved_midpoints = []
    for block_index, rgb_midpoint in enumerate(rgb_midpoints):
        interleaved_midpoints.append(rgb_midpoint)
        if block_index < len(residual_midpoints):
            interleaved_midpoints.append(residual_midpoints[block_index])
    temporal_midpoints = torch.tensor(interleaved_midpoints, dtype=torch.float32)

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    ).strip()
    vision_start = getattr(processor, "vision_start_token", "<|vision_start|>")
    vision_end = getattr(processor, "vision_end_token", "<|vision_end|>")
    video_token = getattr(processor, "video_token", "<|video_pad|>")
    original_placeholder = vision_start + video_token + vision_end
    if text.count(original_placeholder) != 1:
        raise ValueError(
            "Chat template must contain exactly one Qwen3-VL video placeholder."
        )
    tokens_per_block = grid_h * grid_w // merge_size**2
    interleaved_placeholder = _build_interleaved_placeholder(
        processor, temporal_midpoints.tolist(), tokens_per_block
    )
    text = text.replace(original_placeholder, interleaved_placeholder, 1)
    tokenized = processor.tokenizer(
        [text], return_tensors="pt", add_special_tokens=False
    )

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "pixel_values_videos": rgb_inputs["pixel_values_videos"],
        "pixel_values_residuals": pixel_values_residuals,
        "rgb_video_grid_thw": rgb_video_grid_thw,
        "residual_grid_thw": residual_grid_thw,
        "video_grid_thw": video_grid_thw,
        "rgb_temporal_midpoints": torch.tensor(rgb_midpoints, dtype=torch.float32),
        "residual_temporal_midpoints": torch.tensor(
            residual_midpoints, dtype=torch.float32
        ),
        "temporal_midpoints": temporal_midpoints,
        "rit_text": text,
    }
