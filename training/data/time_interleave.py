import torch
from qwen_vl_utils import process_vision_info
from transformers.feature_extraction_utils import BatchFeature


def _single_token_id(tokenizer, token):
    token_ids = tokenizer(token, add_special_tokens=False).input_ids
    if len(token_ids) != 1:
        raise ValueError(f"{token!r} must tokenize to exactly one token.")
    return int(token_ids[0])


def build_interleave_video_inputs(
    messages,
    text,
    processor,
    frame_time_token="<TIME_SAMPLE>",
):
    """Build TimeLens-style interleaved image blocks for Qwen2.5-VL TimeEnc.

    Each temporal visual block keeps its two sampled frames. The two sampling
    timestamps are represented by two ``frame_time_token`` placeholders placed
    immediately before that visual block.
    """
    _images, videos, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True, return_video_metadata=True
    )
    if videos is None or len(videos) == 0:
        raise ValueError("Empty videos for interleave path.")

    video_tensor, metadata = videos[0]

    fps = float(metadata["fps"])
    frame_indices = metadata["frames_indices"]
    if hasattr(frame_indices, "tolist"):
        frame_indices = frame_indices.tolist()
    frame_times = [float(idx) / fps for idx in frame_indices]

    video_processor = processor.video_processor
    videos_inputs = video_processor(
        videos=[video_tensor], do_resize=False, **video_kwargs
    )
    grid_t, grid_h, grid_w = [int(x) for x in videos_inputs["video_grid_thw"][0]]
    merge_length = int(video_processor.merge_size) ** 2
    image_pad_count = (grid_h * grid_w) // merge_length

    expected_frame_times = 2 * grid_t
    if expected_frame_times != len(frame_times):
        raise ValueError(
            "interleave frame-time alignment is broken: "
            f"2 * video_grid_thw temporal dim ({expected_frame_times}) != "
            f"frame_times ({len(frame_times)}); "
            f"frames_indices={frame_indices}, grid_t={grid_t}, "
            f"video_tensor.shape[0]={int(video_tensor.shape[0])}."
        )

    image_token = processor.image_token
    video_token = processor.video_token
    vision_start = "<|vision_start|>"
    vision_end = "<|vision_end|>"
    per_block = vision_start + image_token * image_pad_count + vision_end
    sample_prefix = frame_time_token + frame_time_token
    interleaved = "".join(sample_prefix + per_block for _ in range(grid_t))
    target = vision_start + video_token + vision_end
    if target not in text:
        raise ValueError("video placeholder block not found in chat-templated text.")
    new_text = text.replace(target, interleaved, 1)

    tokenizer = processor.tokenizer
    enc = tokenizer([new_text], return_tensors="pt", add_special_tokens=False)

    input_ids = enc["input_ids"][0]
    frame_tid = _single_token_id(tokenizer, frame_time_token)
    image_tid = _single_token_id(tokenizer, image_token)
    video_tid = _single_token_id(tokenizer, video_token)
    vision_start_id = _single_token_id(tokenizer, vision_start)

    if int((input_ids == video_tid).sum().item()) != 0:
        raise ValueError("interleave input_ids must not contain <|video_pad|>.")
    actual_image_pads = int((input_ids == image_tid).sum().item())
    expected_image_pads = grid_t * image_pad_count
    if actual_image_pads != expected_image_pads:
        raise ValueError(
            f"<|image_pad|> count mismatch: {actual_image_pads} != {expected_image_pads}."
        )

    frame_positions = (input_ids == frame_tid).nonzero(as_tuple=True)[0]
    vision_positions = (input_ids == vision_start_id).nonzero(as_tuple=True)[0]
    if frame_positions.numel() != expected_frame_times or vision_positions.numel() != grid_t:
        raise ValueError(
            f"{frame_time_token}, <|vision_start|>, and image_grid_thw counts must "
            f"match: {int(frame_positions.numel())}, {int(vision_positions.numel())}, "
            f"{grid_t}."
        )
    if not torch.equal(frame_positions[0::2] + 2, vision_positions) or not torch.equal(
        frame_positions[1::2] + 1, vision_positions
    ):
        raise ValueError(
            f"Two {frame_time_token} tokens must immediately precede each "
            "<|vision_start|>."
        )

    image_grid_thw = torch.tensor(
        [[1, grid_h, grid_w] for _ in range(grid_t)], dtype=torch.long
    )
    inputs = BatchFeature(
        {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "pixel_values": videos_inputs["pixel_values_videos"],
            "image_grid_thw": image_grid_thw,
        }
    )
    return inputs, new_text, frame_times
