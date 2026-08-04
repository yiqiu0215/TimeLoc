import copy
import json
import math
from pathlib import Path

from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset

from training.data.preprocess import preprocess


BOUNDARY_STATUS_PROMPT = (
    "You are given a video and a target event boundary. "
    "At {timestamp} seconds, focus on the subject: {subject}. "
    "Describe the subject's status immediately before and immediately after "
    "this timestamp. Respond using exactly two lines:\n"
    "Status_Before: <status immediately before the boundary>\n"
    "Status_After: <status immediately after the boundary>"
)


def _format_timestamp(timestamp: float) -> str:
    return f"{timestamp:.6f}".rstrip("0").rstrip(".")


def _format_response(status_before: str, status_after: str) -> str:
    return f"Status_Before: {status_before}\nStatus_After: {status_after}"


class GEBPlusDataset(Dataset):
    def __init__(self, processor, model_args, data_args, training_args):
        super().__init__()
        self.processor = processor
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        annotation_path = Path(data_args.gebplus_annotation_path)
        video_root = Path(data_args.gebplus_video_root)
        if not annotation_path.is_file():
            raise FileNotFoundError(f"GEB+ annotation file not found: {annotation_path}")
        if not video_root.is_dir():
            raise FileNotFoundError(f"GEB+ video root not found: {video_root}")

        with annotation_path.open("r", encoding="utf-8") as file:
            raw_annotations = json.load(file)
        if not isinstance(raw_annotations, dict):
            raise ValueError("GEB+ annotation root must be an object keyed by video ID.")

        annos = []
        missing_videos = []
        boundary_ids = set()
        for video_id, boundaries in raw_annotations.items():
            if not isinstance(boundaries, list) or not boundaries:
                continue

            video_path = video_root / f"{video_id}.mp4"
            if not video_path.is_file():
                missing_videos.append(str(video_path))
                continue

            duration = max(float(item["next_timestamp"]) for item in boundaries)
            if data_args.min_video_len >= 0 and duration < data_args.min_video_len:
                continue
            if data_args.max_video_len >= 0 and duration > data_args.max_video_len:
                continue

            for item in boundaries:
                anno = self._parse_annotation(video_id, video_path, duration, item)
                if anno["boundary_id"] in boundary_ids:
                    raise ValueError(
                        f"Duplicate GEB+ boundary_id: {anno['boundary_id']}"
                    )
                boundary_ids.add(anno["boundary_id"])
                annos.append(anno)

        if missing_videos:
            preview = ", ".join(missing_videos[:5])
            raise FileNotFoundError(
                f"Missing {len(missing_videos)} GEB+ videos under {video_root}. "
                f"Examples: {preview}"
            )
        if not annos:
            raise ValueError("No valid GEB+ training annotations were loaded.")
        self.annos = annos

    @staticmethod
    def _parse_annotation(video_id, video_path, duration, item):
        required_text_fields = (
            "boundary_id",
            "subject",
            "status_before",
            "status_after",
        )
        for field in required_text_fields:
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Invalid GEB+ {field} in video {video_id}: {value!r}")

        timestamp = float(item["timestamp"])
        prev_timestamp = float(item["prev_timestamp"])
        next_timestamp = float(item["next_timestamp"])
        if not all(
            math.isfinite(value)
            for value in (timestamp, prev_timestamp, next_timestamp)
        ):
            raise ValueError(f"Non-finite GEB+ timestamp in {item['boundary_id']}")
        if not 0 <= prev_timestamp <= timestamp <= next_timestamp <= duration:
            raise ValueError(
                f"Invalid timestamp order in {item['boundary_id']}: "
                f"{prev_timestamp} <= {timestamp} <= {next_timestamp} <= {duration}"
            )

        return {
            "source": "GEB+",
            "data_type": "boundary_status",
            "video_id": video_id,
            "video_path": str(video_path),
            "duration": duration,
            "boundary_id": item["boundary_id"].strip(),
            "timestamp": timestamp,
            "prev_timestamp": prev_timestamp,
            "next_timestamp": next_timestamp,
            "label": item.get("label"),
            "subject": item["subject"].strip(),
            "status_before": item["status_before"].strip(),
            "status_after": item["status_after"].strip(),
        }

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):
        anno = copy.deepcopy(self.annos[index])
        video_content = {
            "type": "video",
            "video": anno["video_path"],
            "min_pixels": int(self.data_args.min_tokens * 32 * 32),
            "total_pixels": int(self.data_args.total_tokens * 32 * 32),
            "fps": float(self.data_args.fps),
        }
        if self.data_args.fps_max_frames is not None:
            video_content["max_frames"] = int(self.data_args.fps_max_frames)

        prompt = BOUNDARY_STATUS_PROMPT.format(
            timestamp=_format_timestamp(anno["timestamp"]),
            subject=anno["subject"],
        )
        messages = [
            {
                "role": "user",
                "content": [
                    video_content,
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": _format_response(
                    anno["status_before"], anno["status_after"]
                ),
            },
        ]

        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if videos is None or len(videos) == 0:
            raise ValueError(f"No video was decoded for {anno['boundary_id']}")
        videos, video_metadatas = zip(*videos)

        text = self.processor.apply_chat_template(messages, tokenize=False).strip()
        inputs = self.processor(
            text=[text],
            images=images,
            videos=list(videos),
            video_metadata=list(video_metadatas),
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        )
        inputs["input_ids"] = inputs["input_ids"][0]
        inputs["labels"] = preprocess(
            inputs["input_ids"],
            text,
            self.processor.tokenizer,
            self.model_args.conv_type,
        )
        return inputs
