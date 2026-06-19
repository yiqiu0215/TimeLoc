import copy
import random
from pathlib import Path

import nncore
import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset

from timelens.dataset.timelens_data import TimeLens100KDataset, parse_query
from training.data.preprocess import IGNORE_INDEX, preprocess
from training.data.time_interleave import build_interleave_video_inputs

GROUNDING_PROMPT = (
    "Please find the visual event described by the sentence '{}', determining its starting and ending times. "
    "The format should be: 'The event happens in <start time> - <end time> seconds'."
)

# prompt for Qwen2.5-VL TimeLens models with interleaved textual timestamps
GROUNDING_PROMPT_TEXT_TIMESTAMP = (
    "You are given a video with multiple frames. "
    "The numbers before each video frame indicate its sampling timestamp (in seconds). "
) + GROUNDING_PROMPT

# prompt for the TimeEnc prefix path: frame timestamps are injected as
# continuous embeddings before the video, not as text numbers.
GROUNDING_PROMPT_TIME_ENC_PREFIX = (
    "You are given a video with multiple frames. "
    "Before the video, a sequence of special time tokens encodes each frame's sampling timestamp (in seconds). "
) + GROUNDING_PROMPT

GROUNDING_PROMPT_TIME_ENC_INTERLEAVE = (
    "You are given a video as a sequence of visual blocks. "
    "Each visual block is preceded by a special time token that encodes the sampling timestamp of that visual block in seconds. "
) + GROUNDING_PROMPT

AUDIO_QUERY_KEYWORDS = {
    "hear",
    "heard",
    "hears",
    "hearing",
    "sound",
    "sounded",
    "sounds",
    "sounding",
    "audio",
}


def _is_audio_related_query(query: str) -> bool:
    words = query.strip("?").lower().split()
    return any(keyword in words for keyword in AUDIO_QUERY_KEYWORDS)


def _normalize_spans(span):
    if isinstance(span, tuple):
        return [list(span)]
    if isinstance(span, list) and len(span) > 0 and isinstance(span[0], (list, tuple)):
        return [list(s) for s in span]
    if isinstance(span, list) and len(span) == 2 and isinstance(span[0], (int, float)):
        return [span]
    raise ValueError(f"Unsupported span format: {span}")


def _format_response(spans):
    return (
        "The event happens in "
        + ", ".join([f"{s:.1f} - {e:.1f} seconds" for s, e in spans])
        + "."
    )


def _format_response_time_stamp(spans, time_stamp_token="<TIME_STAMP>"):
    """Answer template for the time distribution head: one placeholder per span.

    The actual seconds are NOT written into the text; they are supervised by the
    time head via the structured ``time_gt`` field instead.
    """
    return (
        "The event happens in "
        + ", ".join([time_stamp_token for _ in spans])
        + "."
    )


def _extract_qwen2_timelens_sampled_timestamps(videos):
    if videos is None or len(videos) == 0:
        raise ValueError("Expected non-empty videos for Qwen2.5-TimeLens strict path.")
    if not isinstance(videos[0], (list, tuple)) or len(videos[0]) != 2:
        raise ValueError(
            "Qwen2.5-TimeLens strict path expects videos to contain "
            "(video_tensor, metadata) tuples."
        )

    metadata = videos[0][1]
    fps = float(metadata["fps"])
    frame_indices = metadata["frames_indices"]
    if hasattr(frame_indices, "tolist"):
        frame_indices = frame_indices.tolist()
    return [float(idx) / fps for idx in frame_indices[::2]]


def _align_spans_to_sampled_timestamps(spans, sampled_timestamps):
    aligned_spans = []
    for start, end in spans:
        start_idx = 0
        for i, cur_ts in enumerate(sampled_timestamps):
            if cur_ts <= start:
                start_idx = i
            else:
                break

        end_idx = len(sampled_timestamps) - 1
        for i in range(start_idx, len(sampled_timestamps)):
            if end <= sampled_timestamps[i]:
                end_idx = i
                break

        aligned_spans.append([sampled_timestamps[start_idx], sampled_timestamps[end_idx]])
    return aligned_spans


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


def _build_video_content(anno, data_args, include_video_range=False, model_path=None):
    # Qwen2.5-VL / Qwen2.5-TimeLens uses 28x28; Qwen3-VL / TimeLens-8B uses 32x32
    is_qwen2_family = _is_qwen2_model(model_path or "") or _is_qwen2_timelens_model(
        model_path or ""
    )
    scale = 28 * 28 if is_qwen2_family else 32 * 32
    content = {
        "type": "video",
        "video": anno["video_path"],
        "min_pixels": int(data_args.min_tokens * scale),
        "total_pixels": int(data_args.total_tokens * scale),
        "fps": float(data_args.fps),
    }
    if include_video_range:
        content["video_start"] = anno.get("video_start")
        content["video_end"] = anno.get("video_end")
    if getattr(data_args, "fps_max_frames", None) is not None:
        content["max_frames"] = int(data_args.fps_max_frames)
    return content


def _load_filtered_annos(path: str):
    loaded = nncore.load(path)
    if isinstance(loaded, dict):
        loaded = [loaded]
    if loaded is None:
        return []
    annos = []
    for raw in loaded:
        if "source" not in raw or "query" not in raw:
            continue
        annos.append(
            {
                "source": raw["source"],
                "data_type": raw.get("data_type", "grounding"),
                "video_path": raw["video_path"],
                "duration": raw["duration"],
                "query": parse_query(raw["query"]),
                "span": raw["span"],
                "iou": raw.get("iou"),
                "pred": raw.get("pred"),
                "answer": raw.get("answer"),
            }
        )
    return annos


class GroundingDataset(Dataset):
    def __init__(
        self,
        processor,
        model_args,
        data_args,
        training_args,
        dataset_name: str,
        filter_args=None,
        training_mode: str = "sft",
    ):
        super().__init__()
        self.processor = processor
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.training_mode = training_mode
        self._format_model_path = (
            model_args.processor_path or model_args.model_name_or_path or ""
        )
        self._is_qwen2_timelens = _is_qwen2_timelens_model(self._format_model_path)
        self._is_qwen2 = _is_qwen2_model(self._format_model_path)

        self._enable_time_dist = getattr(model_args, "enable_time_dist", False)
        self._time_stamp_token = getattr(
            model_args, "time_stamp_token", "<TIME_STAMP>"
        )
        self._frame_time_token = getattr(
            model_args, "frame_time_token", "<FRAME_TIME>"
        )
        self._time_enc_layout = getattr(model_args, "time_enc_layout", "prefix")
        if self._time_enc_layout not in ("prefix", "interleave"):
            raise ValueError(
                "time_enc_layout must be either 'prefix' or 'interleave', "
                f"got {self._time_enc_layout!r}."
            )
        if self._enable_time_dist and not self._is_qwen2:
            # TimeEnc requires the standard Qwen2.5-VL processor (continuous frame
            # times), not the TimeLens-7B textual-timestamp processor. The run
            # script auto-switches processor_path; fail loudly if it didn't.
            raise ValueError(
                "enable_time_dist requires a standard Qwen2.5-VL processor "
                f"(got processor_path resolving to {self._format_model_path!r}). "
                "Do not use the TimeLens-7B processor with TimeEnc."
            )

        if dataset_name in ("gemini_refined_data", "timelens-100k"):
            base_annos = TimeLens100KDataset.load_annos(split="train")
            if dataset_name == "gemini_refined_data":
                raw_annos = [
                    anno
                    for anno in base_annos
                    if not _is_audio_related_query(anno["query"])
                ]
            else:
                raw_annos = base_annos
        elif dataset_name == "filtered_hybrid":
            if not data_args.raw_anno_path:
                raise ValueError(
                    "raw_anno_path is required for filtered_hybrid dataset."
                )
            if not Path(data_args.raw_anno_path).exists():
                raise FileNotFoundError(
                    f"raw_anno_path does not exist: {data_args.raw_anno_path}"
                )
            raw_annos = _load_filtered_annos(data_args.raw_anno_path)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        annos = []
        for anno in raw_annos:
            num_words = len(anno["query"].split(" "))
            if data_args.min_num_words >= 0 and num_words < data_args.min_num_words:
                continue
            if data_args.max_num_words >= 0 and num_words > data_args.max_num_words:
                continue
            if (
                data_args.min_video_len >= 0
                and anno.get("duration", float("inf")) < data_args.min_video_len
            ):
                continue
            if (
                data_args.max_video_len >= 0
                and anno.get("duration", 0) > data_args.max_video_len
            ):
                continue
            duration = anno.get("duration")
            spans = _normalize_spans(anno["span"])
            if duration and not any(0 <= s <= e <= duration for s, e in spans):
                continue
            anno = dict(anno)
            anno["span"] = spans
            annos.append(anno)

        if filter_args is not None:
            annos = self._filter_annos(annos, filter_args)

        self.annos = annos
        self.raw_length = len(raw_annos)

    def _filter_annos(self, annos, filter_args):
        unique_videos = filter_args.get("unique_videos", False)
        if unique_videos:
            seen = set()
            uniq = []
            for anno in annos:
                vpath = anno["video_path"]
                if vpath in seen:
                    continue
                seen.add(vpath)
                uniq.append(anno)
            annos = uniq

        filter_ratio = filter_args.get("filter_ratio")
        filter_target_size = filter_args.get("filter_target_size")
        if filter_ratio is None and filter_target_size is None:
            return annos

        gaussian_filter_mean = getattr(self.data_args, "gaussian_filter_mean", None)
        gaussian_filter_std = getattr(self.data_args, "gaussian_filter_std", None)
        if (gaussian_filter_mean is None) != (gaussian_filter_std is None):
            raise ValueError(
                "gaussian_filter_mean and gaussian_filter_std should be provided together."
            )
        if gaussian_filter_mean is not None and not annos:
            return annos
        if gaussian_filter_mean is not None and "iou" not in annos[0]:
            raise ValueError("Gaussian filtering requires 'iou' in annotations.")

        seed = getattr(self.training_args, "seed", 42)
        rng = np.random.default_rng(seed)
        py_rng = random.Random(seed)

        buckets = {duration_range: [] for duration_range in filter_args["filter_range"]}
        kept_indices = []
        for idx, anno in enumerate(annos):
            matched = False
            for duration_range in buckets:
                min_duration, max_duration = duration_range
                if min_duration <= anno["duration"] <= max_duration:
                    buckets[duration_range].append(idx)
                    matched = True
                    break
            if not matched:
                kept_indices.append(idx)

        for i, (duration_range, indices) in enumerate(buckets.items()):
            if len(indices) == 0:
                continue
            num_to_select = (
                int(len(indices) * filter_ratio[i])
                if filter_ratio is not None
                else int(filter_target_size[i])
            )
            num_to_select = min(num_to_select, len(indices))

            if gaussian_filter_mean is not None:
                iou_list = np.array(
                    [annos[idx]["iou"] for idx in indices], dtype=np.float64
                )
                weights = np.exp(
                    -0.5
                    * ((iou_list - gaussian_filter_mean) / gaussian_filter_std) ** 2
                )
                if getattr(self.data_args, "fixed_gaussian_sampling", False):
                    num_bins = 20
                    counts, bin_edges = np.histogram(
                        iou_list, bins=num_bins, range=(0, 1)
                    )
                    bin_indices = np.digitize(iou_list, bins=bin_edges)
                    bin_indices = np.clip(bin_indices, 1, num_bins) - 1
                    inverse_density = 1.0 / (counts + 1e-6)
                    weights *= inverse_density[bin_indices]
                weights = weights / weights.sum()
                selected_indices = rng.choice(
                    indices, size=num_to_select, replace=False, p=weights
                ).tolist()
            else:
                selected_indices = py_rng.sample(indices, num_to_select)
            kept_indices.extend(selected_indices)

        return [annos[i] for i in range(len(annos)) if i in kept_indices]

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        if self.training_mode == "sft":
            return self._getitem_sft(idx)
        if self.training_mode == "grpo":
            return self._getitem_grpo(idx)
        raise ValueError(f"Unsupported training_mode: {self.training_mode}")

    def _getitem_sft(self, idx):
        anno = copy.deepcopy(self.annos[idx])
        spans = _normalize_spans(anno["span"])
        # TimeEnc path: standard Qwen2.5-VL processor (continuous frame-time
        # embeddings) instead of TimeLens textual timestamps.
        use_time_enc = self._enable_time_dist and self._is_qwen2
        use_time_enc_interleave = (
            use_time_enc and self._time_enc_layout == "interleave"
        )
        if use_time_enc:
            prompt = (
                GROUNDING_PROMPT_TIME_ENC_INTERLEAVE
                if use_time_enc_interleave
                else GROUNDING_PROMPT_TIME_ENC_PREFIX
            )
        elif self._is_qwen2_timelens:
            prompt = GROUNDING_PROMPT_TEXT_TIMESTAMP
        else:
            prompt = GROUNDING_PROMPT

        messages = [
            {
                "role": "user",
                "content": [
                    _build_video_content(
                        anno, self.data_args, model_path=self._format_model_path
                    ),
                    {"type": "text", "text": prompt.format(anno["query"])},
                ],
            }
        ]

        if use_time_enc_interleave:
            response = _format_response_time_stamp(spans, self._time_stamp_token)
            messages.append({"role": "assistant", "content": response})
            text = self.processor.apply_chat_template(messages, tokenize=False).strip()
            inputs, new_text, sampled_timestamps = build_interleave_video_inputs(
                messages,
                text,
                self.processor,
                frame_time_token=self._frame_time_token,
            )
            spans = _align_spans_to_sampled_timestamps(spans, sampled_timestamps)
            inputs["input_ids"] = inputs["input_ids"][0]
            inputs["labels"] = preprocess(
                inputs["input_ids"],
                new_text,
                self.processor.tokenizer,
                self.model_args.conv_type,
            )
            inputs["time_gt"] = [[float(s), float(e)] for s, e in spans]
            inputs["duration"] = float(anno["duration"])
            inputs["frame_times"] = [float(t) for t in sampled_timestamps]
            return inputs

        if self._is_qwen2_timelens:
            images, videos = process_vision_info(
                messages,
                return_video_metadata=True,
            )
            if videos is None or len(videos) == 0:
                raise ValueError(
                    "Empty videos for Qwen2.5-TimeLens strict path. "
                    "Please ensure processor/config and qwen_vl_utils are aligned."
                )
            video_metadatas = None
            spans = _align_spans_to_sampled_timestamps(
                spans,
                _extract_qwen2_timelens_sampled_timestamps(videos),
            )
        elif self._is_qwen2:
            if use_time_enc:
                # Need video metadata to derive per-grid sampling timestamps.
                images, videos, video_kwargs = process_vision_info(
                    messages,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                )
                if videos is None or len(videos) == 0:
                    raise ValueError("Empty videos for Qwen2.5-VL TimeEnc path.")
                videos_with_meta = videos
                videos = [v[0] for v in videos_with_meta]
                sampled_timestamps = _extract_qwen2_timelens_sampled_timestamps(
                    videos_with_meta
                )
                spans = _align_spans_to_sampled_timestamps(spans, sampled_timestamps)
            else:
                images, videos, video_kwargs = process_vision_info(
                    messages,
                    return_video_kwargs=True,
                )
            video_metadatas = None
        else:
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None

        if self._enable_time_dist:
            response = _format_response_time_stamp(spans, self._time_stamp_token)
        else:
            response = _format_response(spans)
        messages.append({"role": "assistant", "content": response})

        text = self.processor.apply_chat_template(messages, tokenize=False)
        text = [text.strip()]
        if self._is_qwen2_timelens:
            inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                return_tensors="pt",
                do_resize=False,
            )
        elif self._is_qwen2:
            inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                return_tensors="pt",
                do_resize=False,
                **video_kwargs,
            )
        else:
            inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                return_tensors="pt",
                do_resize=False,
                **video_kwargs,
            )
        inputs["input_ids"] = inputs["input_ids"][0]
        inputs["labels"] = preprocess(
            inputs["input_ids"],
            text[0],
            self.processor.tokenizer,
            self.model_args.conv_type,
        )
        if self._enable_time_dist:
            # spans here are already aligned to the sampled timestamps (seconds),
            # matching what the baseline text answer would have targeted.
            inputs["time_gt"] = [[float(s), float(e)] for s, e in spans]
            inputs["duration"] = float(anno["duration"])
        if use_time_enc:
            # Splice <FRAME_TIME> placeholders before the video block and record
            # the per-grid sampling timestamps for TimeEnc injection.
            inputs["frame_times"] = [float(t) for t in sampled_timestamps]
            self._splice_frame_time_tokens(inputs, len(sampled_timestamps))
        return inputs

    def _splice_frame_time_tokens(self, inputs, num_frames):
        """Insert ``num_frames`` <FRAME_TIME> ids right before <|vision_start|>.

        <FRAME_TIME> are pure text tokens placed outside the contiguous video
        block, so Qwen2.5-VL's MRoPE / vision scatter are unaffected. Labels get
        IGNORE_INDEX at the spliced positions (they sit in the user turn).
        """
        tokenizer = self.processor.tokenizer
        frame_tid = tokenizer.convert_tokens_to_ids(self._frame_time_token)
        vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")

        input_ids = inputs["input_ids"]
        labels = inputs["labels"]

        grid_t = int(inputs["video_grid_thw"][0][0])
        if grid_t != num_frames:
            raise ValueError(
                f"video_grid_thw temporal dim ({grid_t}) != sampled_timestamps "
                f"({num_frames}); frame-time alignment is broken."
            )

        positions = (input_ids == vision_start_id).nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            raise ValueError("No <|vision_start|> in input_ids; cannot splice <FRAME_TIME>.")
        p = int(positions[0].item())

        prefix = torch.full((num_frames,), int(frame_tid), dtype=input_ids.dtype)
        ignore = torch.full((num_frames,), IGNORE_INDEX, dtype=labels.dtype)
        inputs["input_ids"] = torch.cat([input_ids[:p], prefix, input_ids[p:]])
        inputs["labels"] = torch.cat([labels[:p], ignore, labels[p:]])

    def _getitem_grpo(self, idx):
        anno = copy.deepcopy(self.annos[idx])
        prompt = (
            GROUNDING_PROMPT_TEXT_TIMESTAMP
            if self._is_qwen2_timelens
            else GROUNDING_PROMPT
        )

        messages = [
            {
                "role": "user",
                "content": [
                    _build_video_content(
                        anno,
                        self.data_args,
                        include_video_range=True,
                        model_path=self._format_model_path,
                    ),
                    {"type": "text", "text": prompt.format(anno["query"])},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        text = [text]

        if self._is_qwen2_timelens:
            images, videos = process_vision_info(
                messages,
                return_video_metadata=True,
            )
            if videos is None or len(videos) == 0:
                raise ValueError(
                    "Empty videos for Qwen2.5-TimeLens strict path. "
                    "Please ensure processor/config and qwen_vl_utils are aligned."
                )
            video_metadatas = None
            anno["span"] = _align_spans_to_sampled_timestamps(
                _normalize_spans(anno["span"]),
                _extract_qwen2_timelens_sampled_timestamps(videos),
            )
        elif self._is_qwen2:
            images, videos, video_kwargs = process_vision_info(
                messages,
                return_video_kwargs=True,
            )
            video_metadatas = None
        else:
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None

        if self._is_qwen2_timelens:
            inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                return_tensors="pt",
                do_resize=False,
            )
        elif self._is_qwen2:
            inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                return_tensors="pt",
                do_resize=False,
                **video_kwargs,
            )
        else:
            inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                return_tensors="pt",
                do_resize=False,
                **video_kwargs,
            )
        inputs["input_ids"] = inputs["input_ids"][0]
        inputs["prompt"] = messages
        inputs["prompt_text"] = text[0]
        inputs["anno"] = anno
        return inputs
