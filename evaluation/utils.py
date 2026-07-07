# Copyright (c) 2025 Jun Zhang. Licensed under the BSD-3-Clause License.

import copy

import torch
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset

from training.data.grounding import _extract_qwen2_timelens_sampled_timestamps
from training.data.time_interleave import build_interleave_video_inputs

GROUNDER_PROMPT = (
    "Please find the visual event described by the sentence '{}', determining its starting and ending times. "
    "The format should be: 'The event happens in <start time> - <end time> seconds'."
)

# prompt for Qwen2.5-VL TimeLens models with interleaved textual timestamps
GROUNDER_PROMPT_TEXT_TIMESTAMP = (
    "You are given a video with multiple frames. "
    "The numbers before each video frame indicate its sampling timestamp (in seconds). "
) + GROUNDER_PROMPT

# prompt for the TimeEnc prefix path: frame timestamps injected as continuous
# embeddings before the video, not text numbers.
GROUNDER_PROMPT_TIME_ENC_PREFIX = (
    "You are given a video with multiple frames. "
    "Before the video, a sequence of special time tokens encodes each frame's sampling timestamp (in seconds). "
) + GROUNDER_PROMPT

GROUNDER_PROMPT_TIME_ENC_INTERLEAVE = (
    "You are given a video as a sequence of visual blocks. "
    "Each visual block is preceded by two special interval time tokens that encode the sampling timestamps of the two frames in that visual block in seconds. "
) + GROUNDER_PROMPT


def _is_qwen2_timelens_model(model_path: str) -> bool:
    if not model_path:
        return False
    m = model_path.lower()
    return "timelens-3b" in m or "timelens-7b" in m


def _is_qwen2_model(model_path: str) -> bool:
    if not model_path:
        return False
    m = model_path.lower()
    return "qwen2" in m or "qwen2.5-vl" in m or "qwen2.5_vl" in m


def _is_qwen3_model(model_path: str) -> bool:
    if not model_path:
        return False
    m = model_path.lower()
    return "qwen3" in m or "timelens-2b" in m or "timelens-8b" in m


class GroundingDataset(Dataset):
    def __init__(self, annos, processor, args):
        super().__init__()
        self.annos = annos
        self.processor = processor
        self.args = args
        self._format_model_path = (
            getattr(args, "format_model_path", None)
            or getattr(args, "processor_path", None)
            or args.model_path
        )
        self._is_qwen2_timelens = _is_qwen2_timelens_model(self._format_model_path)
        self._is_qwen2 = _is_qwen2_model(self._format_model_path)
        self._is_qwen3 = _is_qwen3_model(self._format_model_path)
        self._enable_time_dist = getattr(args, "enable_time_dist", False)
        self._frame_time_token = getattr(args, "frame_time_token", "<TIME_SAMPLE>")
        self._time_enc_layout = getattr(args, "time_enc_layout", "prefix")
        if self._time_enc_layout not in ("prefix", "interleave"):
            raise ValueError(
                "time_enc_layout must be either 'prefix' or 'interleave', "
                f"got {self._time_enc_layout!r}."
            )
        # Routed by the flag, NOT the path string: at eval the processor_path is
        # the training output dir (e.g. ".../TimeLens-3B/...") which would falsely
        # match the timelens path. enable_time_dist implies the standard Qwen2.5-VL
        # processor (saved with the checkpoint) + TimeEnc.
        self._use_time_enc = bool(self._enable_time_dist)
        if self._use_time_enc:
            # TimeEnc: continuous frame-time embeddings, standard Qwen2.5-VL processor.
            self.prompt = (
                GROUNDER_PROMPT_TIME_ENC_INTERLEAVE
                if self._time_enc_layout == "interleave"
                else GROUNDER_PROMPT_TIME_ENC_PREFIX
            )
        elif self._is_qwen2_timelens:
            # Qwen2.5-TimeLens uses interleaved textual timestamps.
            self.prompt = GROUNDER_PROMPT_TEXT_TIMESTAMP
        else:
            self.prompt = GROUNDER_PROMPT

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):
        anno = copy.deepcopy(self.annos[index])

        video_path = anno["video_path"]
        query = anno["query"]

        if self._is_qwen3:
            # for TimeLens-8B(based on Qwen3-VL) and Qwen3-VL models
            downsample_rate = 32
        elif self._is_qwen2 or self._is_qwen2_timelens:
            # for Qwen2.5-TimeLens and Qwen2.5-VL models
            downsample_rate = 28
        else:
            raise NotImplementedError(
                f"Model {self._format_model_path} not supported yet."
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "min_pixels": self.args.min_tokens * downsample_rate * downsample_rate,
                        "total_pixels": self.args.total_tokens * downsample_rate * downsample_rate,
                        "fps": self.args.fps,
                    },
                    {"type": "text", "text": self.prompt.format(query)},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if self._use_time_enc:
            # TimeEnc: standard Qwen2.5-VL processing + <TIME_SAMPLE> splice.
            # Routed first so the output-dir path string can't misroute to the
            # textual-timestamp path.
            if self._time_enc_layout == "interleave":
                inputs, _new_text, frame_times = build_interleave_video_inputs(
                    messages,
                    text,
                    self.processor,
                    frame_time_token=self._frame_time_token,
                )
                return {"inputs": inputs, "anno": anno, "frame_times": frame_times}

            images, videos, video_kwargs = process_vision_info(
                messages,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            if videos is None or len(videos) == 0:
                raise ValueError("Empty videos for Qwen2.5-VL TimeEnc path.")
            videos_with_meta = videos
            videos = [v[0] for v in videos_with_meta]
            frame_times = _extract_qwen2_timelens_sampled_timestamps(videos_with_meta)
            inputs = self.processor(
                text=[text],
                images=images,
                videos=videos,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
            self._splice_frame_time_tokens(inputs, len(frame_times))
            return {"inputs": inputs, "anno": anno, "frame_times": frame_times}
        elif self._is_qwen2_timelens:
            # for Qwen2.5-TimeLens with interleaved textual timestamps
            images, videos = process_vision_info(messages, return_video_metadata=True)
            inputs = self.processor(
                text=[text],
                images=images,
                videos=videos,
                padding=True,
                return_tensors="pt",
            )
        elif self._is_qwen3:
            # for TimeLens-8B(based on Qwen3-VL) and Qwen3-VL models
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
            inputs = self.processor(
                text=[text],
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
        elif self._is_qwen2:
            # for Qwen2.5-VL model
            images, videos, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )
            inputs = self.processor(
                text=[text],
                images=images,
                videos=videos,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
        else:
            raise NotImplementedError(
                f"Model {self._format_model_path} not supported yet."
            )

        return {"inputs": inputs, "anno": anno}

    def _splice_frame_time_tokens(self, inputs, num_frames):
        """Insert ``num_frames`` <TIME_SAMPLE> ids before <|vision_start|> ([1, L]).

        Mirrors the training-side splice (see grounding.py). MRoPE-safe because
        <TIME_SAMPLE> sit outside the contiguous video block.
        """
        tokenizer = self.processor.tokenizer
        frame_tid = tokenizer.convert_tokens_to_ids(self._frame_time_token)
        vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")

        input_ids = inputs["input_ids"]  # [1, L]
        attn = inputs["attention_mask"]
        grid_t = int(inputs["video_grid_thw"][0][0])
        if grid_t != num_frames:
            raise ValueError(
                f"video_grid_thw temporal dim ({grid_t}) != frame_times ({num_frames})."
            )
        pos = (input_ids[0] == vision_start_id).nonzero(as_tuple=True)[0]
        if pos.numel() == 0:
            raise ValueError("No <|vision_start|> in input_ids; cannot splice frame-time token.")
        p = int(pos[0].item())

        prefix = torch.full(
            (1, num_frames), int(frame_tid), dtype=input_ids.dtype, device=input_ids.device
        )
        ones = torch.ones((1, num_frames), dtype=attn.dtype, device=attn.device)
        inputs["input_ids"] = torch.cat([input_ids[:, :p], prefix, input_ids[:, p:]], dim=1)
        inputs["attention_mask"] = torch.cat([attn[:, :p], ones, attn[:, p:]], dim=1)
