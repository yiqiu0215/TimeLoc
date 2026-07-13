from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

from training.modeling.special_tokens import TIME_BIN_COUNT


@dataclass
class PackedTimeTokenInputs:
    packed_inputs_embeds: torch.Tensor
    packed_attention_mask: torch.Tensor
    packed_position_ids: torch.Tensor
    packed_input_ids: torch.Tensor
    visual_block_token_ranges: list[list[tuple[int, int]]]
    time_token_positions: list[list[int]]
    visual_block_token_counts: list[list[int]]
    packed_labels: Optional[torch.Tensor] = None


class TimeTokenPacker:
    """Pack one video placeholder into temporal visual blocks plus time tokens.

    The processor still supplies one ``<|vision_start|><|video_pad|>
    <|vision_end|>`` span per sample.  This interface replaces that span after
    the vision encoder has produced temporal-patch-merged embeddings:

    ``<vision_start> V_i <vision_end> <time_q_i>``.

    ``packed_position_ids`` defaults to sequential positions for a generic
    language model.  Qwen2.5-VL callers should provide a position-id builder in
    the next integration stage so its 3D M-RoPE positions are computed from the
    packed image-token layout.
    """

    def __init__(
        self,
        embedding_layer,
        vision_start_token_id: int,
        vision_end_token_id: int,
        image_token_id: int,
        video_token_id: int,
        time_token_ids,
        spatial_merge_size: int = 2,
        pad_token_id: int = 0,
        position_id_builder: Optional[Callable] = None,
    ):
        if embedding_layer is None:
            raise ValueError("TimeTokenPacker requires the base input embedding layer.")
        time_token_ids = tuple(int(token_id) for token_id in time_token_ids)
        if not time_token_ids:
            raise ValueError("time_token_ids must be non-empty.")
        if any(
            time_token_ids[index] != time_token_ids[0] + index
            for index in range(len(time_token_ids))
        ):
            raise ValueError("time_token_ids must be contiguous.")
        spatial_merge_size = int(spatial_merge_size)
        if spatial_merge_size <= 0:
            raise ValueError("spatial_merge_size must be positive.")

        self.embedding_layer = embedding_layer
        self.vision_start_token_id = int(vision_start_token_id)
        self.vision_end_token_id = int(vision_end_token_id)
        self.image_token_id = int(image_token_id)
        self.video_token_id = int(video_token_id)
        self.time_token_ids = time_token_ids
        self.spatial_merge_size = spatial_merge_size
        self.pad_token_id = int(pad_token_id)
        self.position_id_builder = position_id_builder

    @staticmethod
    def _normalize_grid(video_grid_thw: torch.Tensor, batch_size: int) -> torch.Tensor:
        grid = torch.as_tensor(video_grid_thw, dtype=torch.long)
        if grid.ndim == 1:
            if batch_size != 1 or grid.numel() != 3:
                raise ValueError(
                    f"video_grid_thw must be [B, 3], got {tuple(grid.shape)}."
                )
            grid = grid.reshape(1, 3)
        if grid.ndim != 2 or grid.shape != (batch_size, 3):
            raise ValueError(
                f"TimeRefine expects one video per sample with video_grid_thw [B, 3], "
                f"got {tuple(grid.shape)} for batch size {batch_size}."
            )
        if torch.any(grid <= 0):
            raise ValueError("video_grid_thw entries must be positive.")
        return grid

    def _split_visual_embeddings(
        self,
        visual_embeddings: torch.Tensor,
        video_grid_thw: torch.Tensor,
    ) -> list[torch.Tensor]:
        grid = video_grid_thw
        counts = []
        for temporal, height, width in grid.tolist():
            if height % self.spatial_merge_size != 0 or width % self.spatial_merge_size != 0:
                raise ValueError(
                    "video_grid_thw height/width must be divisible by spatial_merge_size: "
                    f"grid={(temporal, height, width)}, merge={self.spatial_merge_size}."
                )
            per_block = (height // self.spatial_merge_size) * (
                width // self.spatial_merge_size
            )
            counts.append(int(temporal) * per_block)

        if isinstance(visual_embeddings, (list, tuple)):
            if len(visual_embeddings) != len(counts):
                raise ValueError("One visual embedding tensor is required per sample.")
            samples = [torch.as_tensor(value) for value in visual_embeddings]
            for sample, expected in zip(samples, counts):
                if sample.ndim != 2 or sample.shape[0] != expected:
                    raise ValueError(
                        f"Visual embedding shape must be [{expected}, H], got "
                        f"{tuple(sample.shape)}."
                    )
            return samples

        embeddings = torch.as_tensor(visual_embeddings)
        if embeddings.ndim == 3:
            if embeddings.shape[0] != len(counts):
                raise ValueError("Batched visual embeddings must have batch dimension B.")
            samples = [embeddings[index, :count] for index, count in enumerate(counts)]
            if any(sample.shape[0] != count for sample, count in zip(samples, counts)):
                raise ValueError("Batched visual embeddings do not contain all visual tokens.")
            return samples
        if embeddings.ndim != 2:
            raise ValueError(
                f"visual_embeddings must be [sum(P_i), H] or a per-sample list, "
                f"got {tuple(embeddings.shape)}."
            )
        if embeddings.shape[0] != sum(counts):
            raise ValueError(
                "Visual embedding count does not match video_grid_thw: "
                f"got {embeddings.shape[0]}, expected {sum(counts)}."
            )
        return list(torch.split(embeddings, counts, dim=0))

    @staticmethod
    def _valid_frame_values(
        frame_bin_ids: torch.Tensor,
        frame_timestamps: torch.Tensor,
        frame_valid_mask: Optional[torch.Tensor],
        batch_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bins = frame_bin_ids[batch_index].reshape(-1).to(dtype=torch.long)
        timestamps = frame_timestamps[batch_index].reshape(-1).to(dtype=torch.float32)
        if frame_valid_mask is None:
            valid = torch.ones_like(bins, dtype=torch.bool)
        else:
            valid = frame_valid_mask[batch_index].reshape(-1).to(dtype=torch.bool)
        if bins.numel() != timestamps.numel() or bins.numel() != valid.numel():
            raise ValueError("frame_bin_ids, frame_timestamps and frame_valid_mask lengths differ.")
        bins = bins[valid]
        timestamps = timestamps[valid]
        if bins.numel() == 0:
            raise ValueError("Every sample must contain at least one valid visual block.")
        if torch.any(bins < 0) or torch.any(bins >= TIME_BIN_COUNT):
            raise ValueError("frame_bin_ids must lie in [0, 300].")
        if not torch.isfinite(timestamps).all():
            raise ValueError("frame_timestamps must be finite.")
        if timestamps.numel() > 1 and not torch.all(timestamps[1:] > timestamps[:-1]):
            raise ValueError("frame_timestamps must be strictly increasing.")
        return bins, timestamps

    def _build_position_ids(
        self,
        packed_input_ids: torch.Tensor,
        packed_attention_mask: torch.Tensor,
        packed_image_grid_thw: list[torch.Tensor],
    ) -> torch.Tensor:
        if self.position_id_builder is None:
            positions = torch.zeros_like(packed_input_ids)
            for index in range(packed_input_ids.shape[0]):
                length = int(packed_attention_mask[index].sum().item())
                positions[index, :length] = torch.arange(
                    length, device=packed_input_ids.device, dtype=torch.long
                )
            return positions

        per_sample = []
        for index in range(packed_input_ids.shape[0]):
            length = int(packed_attention_mask[index].sum().item())
            position_ids = self.position_id_builder(
                packed_input_ids[index : index + 1, :length],
                packed_attention_mask[index : index + 1, :length],
                packed_image_grid_thw[index],
            )
            position_ids = torch.as_tensor(position_ids, device=packed_input_ids.device)
            if position_ids.ndim == 3 and position_ids.shape[1] == 1:
                position_ids = position_ids[:, 0]
            elif position_ids.ndim == 2 and position_ids.shape[0] == 1:
                position_ids = position_ids[0]
            if position_ids.ndim not in (1, 2) or position_ids.shape[-1] != length:
                raise ValueError(
                    "position_id_builder must return [L], [R, L] or [R, 1, L]."
                )
            per_sample.append(position_ids)

        max_length = packed_input_ids.shape[1]
        rank = max(position.shape[0] if position.ndim == 2 else 1 for position in per_sample)
        padded = packed_input_ids.new_zeros((len(per_sample), rank, max_length))
        for index, position_ids in enumerate(per_sample):
            if position_ids.ndim == 1:
                position_ids = position_ids.unsqueeze(0)
            padded[index, : position_ids.shape[0], : position_ids.shape[1]] = position_ids
        return padded

    def forward(
        self,
        visual_embeddings: torch.Tensor,
        video_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        frame_bin_ids: torch.Tensor,
        frame_timestamps: torch.Tensor,
        frame_valid_mask: Optional[torch.Tensor] = None,
        input_embeddings: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> PackedTimeTokenInputs:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must both have shape [B, L].")
        batch_size = input_ids.shape[0]
        grid = self._normalize_grid(video_grid_thw, batch_size).to(input_ids.device)
        frame_bin_ids = torch.as_tensor(frame_bin_ids, device=input_ids.device)
        frame_timestamps = torch.as_tensor(frame_timestamps, device=input_ids.device)
        if frame_bin_ids.ndim != 2 or frame_timestamps.ndim != 2:
            raise ValueError("frame_bin_ids and frame_timestamps must have shape [B, N].")
        if frame_bin_ids.shape[0] != batch_size or frame_timestamps.shape[0] != batch_size:
            raise ValueError("Frame metadata batch dimensions must match input_ids.")
        if frame_valid_mask is not None:
            frame_valid_mask = torch.as_tensor(frame_valid_mask, device=input_ids.device)
            if frame_valid_mask.shape != frame_bin_ids.shape:
                raise ValueError("frame_valid_mask must have the same shape as frame_bin_ids.")
        if input_embeddings is None:
            input_embeddings = self.embedding_layer(input_ids)
        if input_embeddings.ndim != 3 or input_embeddings.shape[:2] != input_ids.shape:
            raise ValueError("input_embeddings must have shape [B, L, H].")
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids.")

        special_token_ids = torch.tensor(
            [
                self.vision_start_token_id,
                self.vision_end_token_id,
                *self.time_token_ids,
            ],
            dtype=torch.long,
            device=input_embeddings.device,
        ).unsqueeze(0)
        special_token_embeddings = self.embedding_layer(special_token_ids)[0].to(
            dtype=input_embeddings.dtype
        )

        visual_samples = self._split_visual_embeddings(visual_embeddings, grid.cpu())
        sample_embeds = []
        sample_ids = []
        sample_labels = []
        sample_ranges = []
        sample_time_positions = []
        sample_counts = []
        sample_image_grids = []

        for batch_index in range(batch_size):
            bins, timestamps = self._valid_frame_values(
                frame_bin_ids,
                frame_timestamps,
                frame_valid_mask,
                batch_index,
            )
            temporal_blocks = int(grid[batch_index, 0].item())
            if bins.numel() != temporal_blocks:
                raise ValueError(
                    "The number of valid frame metadata rows must equal the temporal "
                    f"grid size: metadata={bins.numel()}, grid_t={temporal_blocks}."
                )
            active = attention_mask[batch_index].to(dtype=torch.bool)
            ids = input_ids[batch_index][active]
            embeds = input_embeddings[batch_index][active]
            current_labels = labels[batch_index][active] if labels is not None else None
            video_positions = (ids == self.video_token_id).nonzero(as_tuple=True)[0]
            if video_positions.numel() == 0:
                raise ValueError(
                    "TimeRefine requires one expanded <|video_pad|> span per sample."
                )
            expected_video_tokens = int(visual_samples[batch_index].shape[0])
            if video_positions.numel() not in (1, expected_video_tokens):
                raise ValueError(
                    "The <|video_pad|> count must be one legacy placeholder or match "
                    f"the visual token count: got {video_positions.numel()}, "
                    f"expected {expected_video_tokens}."
                )
            first_video_position = int(video_positions[0].item())
            last_video_position = int(video_positions[-1].item())
            if last_video_position - first_video_position + 1 != video_positions.numel():
                raise ValueError("TimeRefine requires one contiguous <|video_pad|> span.")
            start = first_video_position - 1
            end = last_video_position + 1
            if (
                start < 0
                or end >= ids.numel()
                or int(ids[start]) != self.vision_start_token_id
                or int(ids[end]) != self.vision_end_token_id
            ):
                raise ValueError(
                    "The video placeholder span must be enclosed by exactly one "
                    "<|vision_start|> and <|vision_end|>."
                )

            prefix_embeds = embeds[:start]
            suffix_embeds = embeds[end + 1 :]
            prefix_ids = ids[:start]
            suffix_ids = ids[end + 1 :]
            prefix_labels = current_labels[:start] if current_labels is not None else None
            suffix_labels = current_labels[end + 1 :] if current_labels is not None else None

            block_embeds = []
            block_ids = []
            block_labels = []
            ranges = []
            time_positions = []
            image_grid = []
            visual_offset = 0
            packed_cursor = int(prefix_embeds.shape[0])
            height = int(grid[batch_index, 1].item())
            width = int(grid[batch_index, 2].item())
            per_block = (height // self.spatial_merge_size) * (
                width // self.spatial_merge_size
            )
            for block_index in range(temporal_blocks):
                visual_block = visual_samples[batch_index][
                    visual_offset : visual_offset + per_block
                ]
                visual_offset += per_block
                q = int(bins[block_index].item())
                time_id = self.time_token_ids[q]
                start_embed = special_token_embeddings[0:1]
                end_embed = special_token_embeddings[1:2]
                time_embed = special_token_embeddings[2 + q : 3 + q]
                block_embeds.extend([start_embed, visual_block, end_embed, time_embed])
                block_ids.extend(
                    [
                        self.vision_start_token_id,
                        *([self.image_token_id] * per_block),
                        self.vision_end_token_id,
                        time_id,
                    ]
                )
                if current_labels is not None:
                    block_labels.append(
                        torch.full(
                            (per_block + 3,),
                            -100,
                            dtype=current_labels.dtype,
                            device=current_labels.device,
                        )
                    )
                visual_start = packed_cursor + 1
                ranges.append((visual_start, visual_start + per_block))
                time_positions.append(visual_start + per_block + 1)
                packed_cursor += per_block + 3
                image_grid.append(
                    torch.tensor(
                        [1, height, width], dtype=torch.long, device=input_ids.device
                    )
                )

            if visual_offset != visual_samples[batch_index].shape[0]:
                raise ValueError("Visual block slicing did not consume all embeddings.")
            middle_embeds = torch.cat(block_embeds, dim=0)
            middle_ids = torch.tensor(block_ids, dtype=ids.dtype, device=ids.device)
            pieces = [prefix_embeds, middle_embeds, suffix_embeds]
            id_pieces = [prefix_ids, middle_ids, suffix_ids]
            final_embeds = torch.cat(pieces, dim=0)
            final_ids = torch.cat(id_pieces, dim=0)
            if final_embeds.shape[0] != final_ids.shape[0]:
                raise ValueError("Packed embeddings and ids have different sequence lengths.")
            sample_embeds.append(final_embeds)
            sample_ids.append(final_ids)
            sample_ranges.append(ranges)
            sample_time_positions.append(time_positions)
            sample_counts.append([per_block] * temporal_blocks)
            sample_image_grids.append(torch.stack(image_grid, dim=0))
            if current_labels is not None:
                final_labels = torch.cat(
                    [prefix_labels, torch.cat(block_labels, dim=0), suffix_labels], dim=0
                )
                sample_labels.append(final_labels)

        packed_inputs_embeds = pad_sequence(sample_embeds, batch_first=True, padding_value=0.0)
        packed_input_ids = pad_sequence(
            sample_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        packed_attention_mask = torch.zeros(
            packed_input_ids.shape,
            dtype=attention_mask.dtype,
            device=packed_input_ids.device,
        )
        for index, ids in enumerate(sample_ids):
            packed_attention_mask[index, : ids.numel()] = 1
        packed_position_ids = self._build_position_ids(
            packed_input_ids,
            packed_attention_mask,
            sample_image_grids,
        )
        packed_labels = None
        if labels is not None:
            packed_labels = pad_sequence(
                sample_labels,
                batch_first=True,
                padding_value=-100,
            )
        return PackedTimeTokenInputs(
            packed_inputs_embeds=packed_inputs_embeds,
            packed_attention_mask=packed_attention_mask,
            packed_position_ids=packed_position_ids,
            packed_input_ids=packed_input_ids,
            visual_block_token_ranges=sample_ranges,
            time_token_positions=sample_time_positions,
            visual_block_token_counts=sample_counts,
            packed_labels=packed_labels,
        )

    __call__ = forward
