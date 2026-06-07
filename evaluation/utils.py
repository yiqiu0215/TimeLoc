# Copyright (c) 2025 Jun Zhang. Licensed under the BSD-3-Clause License.

import copy

from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset

GROUNDER_PROMPT = (
    "Please find the visual event described by the sentence '{}', determining its starting and ending times. "
    "The format should be: 'The event happens in <start time> - <end time> seconds'."
)

# prompt for Qwen2.5-VL TimeLens models with interleaved textual timestamps
GROUNDER_PROMPT_TEXT_TIMESTAMP = (
    "You are given a video with multiple frames. "
    "The numbers before each video frame indicate its sampling timestamp (in seconds). "
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
    return "qwen3" in m or "timelens-8b" in m


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
        if self._is_qwen2_timelens:
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

        if self._is_qwen2_timelens:
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
