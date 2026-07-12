import warnings

import torch
from torch.nn.utils.rnn import pad_sequence


IGNORE_INDEX = -100


class HybridDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [d["input_ids"] for d in batch]
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        labels = [d["labels"] for d in batch]
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        if input_ids.size() != labels.size():
            raise ValueError(
                f"input_ids and labels must have same shape, got {input_ids.size()} vs {labels.size()}."
            )

        seq_len = input_ids.size(1)
        max_len = self.tokenizer.model_max_length
        if seq_len > max_len:
            warnings.warn(
                f"Input sequence length exceeds tokenizer max length: {seq_len} > {max_len}"
            )
            input_ids = input_ids[:, :max_len]
            labels = labels[:, :max_len]

        data = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids != self.tokenizer.pad_token_id,
        }

        time_refine_keys = (
            "frame_bin_ids",
            "frame_timestamps",
            "frame_labels",
            "frame_valid_mask",
            "gt_start",
            "gt_end",
            "duration",
        )
        has_time_refine_fields = any(key in batch[0] for key in time_refine_keys)
        if has_time_refine_fields:
            for sample in batch:
                missing = [key for key in time_refine_keys if key not in sample]
                if missing:
                    raise ValueError(
                        "TimeRefine batch samples must contain the same metadata fields; "
                        f"missing={missing}."
                    )
                lengths = {
                    key: int(torch.as_tensor(sample[key]).reshape(-1).numel())
                    for key in time_refine_keys[:4]
                }
                if len(set(lengths.values())) != 1:
                    raise ValueError(
                        "frame_bin_ids, frame_timestamps, frame_labels and "
                        f"frame_valid_mask must have equal lengths, got {lengths}."
                    )

            data["frame_bin_ids"] = pad_sequence(
                [torch.as_tensor(d["frame_bin_ids"], dtype=torch.long).reshape(-1) for d in batch],
                batch_first=True,
                padding_value=0,
            )
            data["frame_timestamps"] = pad_sequence(
                [
                    torch.as_tensor(d["frame_timestamps"], dtype=torch.float32).reshape(-1)
                    for d in batch
                ],
                batch_first=True,
                padding_value=0.0,
            )
            data["frame_labels"] = pad_sequence(
                [torch.as_tensor(d["frame_labels"], dtype=torch.long).reshape(-1) for d in batch],
                batch_first=True,
                padding_value=0,
            )
            data["frame_valid_mask"] = pad_sequence(
                [
                    torch.as_tensor(d["frame_valid_mask"], dtype=torch.bool).reshape(-1)
                    for d in batch
                ],
                batch_first=True,
                padding_value=False,
            )
            for key in ("gt_start", "gt_end", "duration"):
                values = [
                    torch.as_tensor(d[key], dtype=torch.float32).reshape(-1)
                    for d in batch
                ]
                if any(value.numel() != 1 for value in values):
                    raise ValueError(f"{key} must contain exactly one scalar per sample.")
                data[key] = torch.stack(values, dim=0)

        for key in (
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
        ):
            if key in batch[0]:
                data[key] = torch.cat([d[key] for d in batch])

        if "second_per_grid_ts" in batch[0]:
            data["second_per_grid_ts"] = [t for d in batch for t in d["second_per_grid_ts"]]

        return data
