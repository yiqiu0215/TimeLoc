from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.modeling.special_tokens import TIME_BIN_COUNT
from training.modeling.time_proj import TimeProj


class _RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps).to(hidden_states.dtype)
        return normalized * self.weight.to(normalized.dtype)


class _SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, inner_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, inner_size)
        self.up_proj = nn.Linear(hidden_size, inner_size)
        self.down_proj = nn.Linear(inner_size, hidden_size)

    def forward(self, hidden_states):
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class _TimeRefineTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        ffn_expansion_ratio: int,
        dropout: float,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "TimeRefine hidden size must be divisible by attention heads: "
                f"hidden={hidden_size}, heads={num_attention_heads}."
            )
        self.attn_norm = _RMSNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.ffn_norm = _RMSNorm(hidden_size)
        self.ffn = _SwiGLU(
            hidden_size,
            int(hidden_size) * int(ffn_expansion_ratio),
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, hidden_states, key_padding_mask):
        normalized = self.attn_norm(hidden_states)
        attended, _ = self.self_attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden_states = hidden_states + self.dropout(attended)
        hidden_states = hidden_states + self.dropout(self.ffn(self.ffn_norm(hidden_states)))
        return hidden_states


def _straight_through_round(values: torch.Tensor) -> torch.Tensor:
    clipped = values.clamp(0.0, float(TIME_BIN_COUNT - 1))
    rounded = clipped.round()
    return clipped + (rounded - clipped).detach()


@dataclass
class TimeRefineHeadOutput:
    pred_start_q: torch.Tensor
    pred_end_q: torch.Tensor
    pred_start_q_continuous: torch.Tensor
    pred_end_q_continuous: torch.Tensor
    start_probs: torch.Tensor
    end_probs: torch.Tensor
    start_logits: torch.Tensor
    end_logits: torch.Tensor


class TimeRefineHead(nn.Module):
    """Shared TimeProj/Transformer with independent start/end scorers."""

    def __init__(
        self,
        llm_hidden_size: int,
        time_hidden_size: int = 512,
        num_attention_heads: int = 8,
        ffn_expansion_ratio: int = 4,
        dropout: float = 0.0,
        classification_embedding_dim: int = 32,
        time_embedding_dim: int = 32,
        branch_embedding_dim: int = 32,
        time_bin_count: int = TIME_BIN_COUNT,
    ):
        super().__init__()
        if int(time_bin_count) != TIME_BIN_COUNT:
            raise ValueError(
                f"TimeRefine requires exactly {TIME_BIN_COUNT} time bins, got {time_bin_count}."
            )
        self.llm_hidden_size = int(llm_hidden_size)
        self.time_hidden_size = int(time_hidden_size)
        self.classification_embedding = nn.Embedding(2, int(classification_embedding_dim))
        self.time_embedding = nn.Embedding(TIME_BIN_COUNT, int(time_embedding_dim))
        self.branch_embedding = nn.Embedding(2, int(branch_embedding_dim))
        self.time_proj = TimeProj(
            self.llm_hidden_size
            + int(classification_embedding_dim)
            + int(time_embedding_dim)
            + int(branch_embedding_dim),
            self.time_hidden_size,
            self.time_hidden_size,
        )
        self.transformer = _TimeRefineTransformerBlock(
            hidden_size=self.time_hidden_size,
            num_attention_heads=int(num_attention_heads),
            ffn_expansion_ratio=int(ffn_expansion_ratio),
            dropout=float(dropout),
        )
        self.start_scorer = nn.Linear(self.time_hidden_size, 1)
        self.end_scorer = nn.Linear(self.time_hidden_size, 1)

    @staticmethod
    def _pack_visual_groups(
        visual_groups: Sequence[Sequence[torch.Tensor]],
        llm_hidden_size: int,
    ):
        if not visual_groups:
            raise ValueError("TimeRefineHead requires a non-empty batch.")
        batch_size = len(visual_groups)
        max_blocks = max(len(group) for group in visual_groups)
        if max_blocks == 0:
            raise ValueError("Every refinement group must contain at least one block.")
        max_spatial = 0
        reference = None
        for group in visual_groups:
            for block in group:
                block = torch.as_tensor(block)
                if block.ndim != 2 or block.shape[-1] != llm_hidden_size:
                    raise ValueError(
                        "Each visual block must have shape [P_i, H_llm], got "
                        f"{tuple(block.shape)}."
                    )
                if block.shape[0] <= 0:
                    raise ValueError("Each visual block must contain at least one token.")
                max_spatial = max(max_spatial, int(block.shape[0]))
                reference = block
        visual = reference.new_zeros(
            (batch_size, max_blocks, max_spatial, llm_hidden_size)
        )
        token_mask = torch.zeros(
            (batch_size, max_blocks, max_spatial),
            dtype=torch.bool,
            device=reference.device,
        )
        block_mask = torch.zeros(
            (batch_size, max_blocks), dtype=torch.bool, device=reference.device
        )
        for batch_index, group in enumerate(visual_groups):
            for block_index, block in enumerate(group):
                block = torch.as_tensor(block, device=reference.device, dtype=reference.dtype)
                spatial = int(block.shape[0])
                visual[batch_index, block_index, :spatial] = block
                token_mask[batch_index, block_index, :spatial] = True
                block_mask[batch_index, block_index] = True
        return visual, token_mask, block_mask

    @staticmethod
    def _pack_metadata(metadata_groups, max_blocks, device, dtype, padding_value=0):
        values = []
        for group in metadata_groups:
            tensor = torch.as_tensor(group, device=device, dtype=dtype).reshape(-1)
            if tensor.numel() == 0:
                raise ValueError("Refinement metadata groups cannot be empty.")
            if tensor.numel() > max_blocks:
                raise ValueError(
                    "Refinement metadata group is longer than its visual group: "
                    f"metadata={tensor.numel()}, visual={max_blocks}."
                )
            values.append(tensor)
        padded = torch.full(
            (len(values), max_blocks),
            padding_value,
            dtype=dtype,
            device=device,
        )
        for batch_index, tensor in enumerate(values):
            padded[batch_index, : tensor.numel()] = tensor
        return padded

    def _forward_branch(
        self,
        visual_groups,
        classification_groups,
        relative_time_groups,
        absolute_time_groups,
        branch_id: int,
        scorer: nn.Linear,
    ):
        visual, token_mask, block_mask = self._pack_visual_groups(
            visual_groups, self.llm_hidden_size
        )
        batch_size, max_blocks, max_spatial, _ = visual.shape
        classification = self._pack_metadata(
            classification_groups,
            max_blocks,
            visual.device,
            torch.long,
        )
        relative_time = self._pack_metadata(
            relative_time_groups,
            max_blocks,
            visual.device,
            torch.long,
        )
        absolute_time = self._pack_metadata(
            absolute_time_groups,
            max_blocks,
            visual.device,
            torch.long,
        )
        if torch.any((classification < 0) | (classification > 1)):
            raise ValueError("classification_group values must be 0 or 1.")
        if torch.any((relative_time < 0) | (relative_time >= TIME_BIN_COUNT)):
            raise ValueError("relative time bins must lie in [0, 300].")
        if torch.any((absolute_time < 0) | (absolute_time >= TIME_BIN_COUNT)):
            raise ValueError("absolute time bins must lie in [0, 300].")

        cls_embed = self.classification_embedding(classification)
        time_embed = self.time_embedding(relative_time)
        branch_ids = torch.full(
            (batch_size, max_blocks),
            int(branch_id),
            dtype=torch.long,
            device=visual.device,
        )
        branch_embed = self.branch_embedding(branch_ids)
        features = torch.cat(
            [
                visual,
                cls_embed.unsqueeze(2).expand(-1, -1, max_spatial, -1),
                time_embed.unsqueeze(2).expand(-1, -1, max_spatial, -1),
                branch_embed.unsqueeze(2).expand(-1, -1, max_spatial, -1),
            ],
            dim=-1,
        )
        projected = self.time_proj(features)
        sequence = projected.reshape(batch_size, max_blocks * max_spatial, -1)
        sequence_mask = token_mask.reshape(batch_size, max_blocks * max_spatial)
        sequence = self.transformer(sequence, key_padding_mask=~sequence_mask)
        sequence = sequence.reshape(batch_size, max_blocks, max_spatial, -1)
        pooled = (
            sequence * token_mask.unsqueeze(-1).to(sequence.dtype)
        ).sum(dim=2) / token_mask.sum(dim=2, keepdim=True).clamp_min(1).to(sequence.dtype)

        logits = scorer(pooled).squeeze(-1)
        logits = logits.masked_fill(~block_mask, torch.finfo(logits.dtype).min)
        probs = torch.softmax(logits.float(), dim=-1)
        absolute_time_float = absolute_time.float()
        expected_q = (probs * absolute_time_float).sum(dim=-1)
        rounded_q = _straight_through_round(expected_q)
        return {
            "q": rounded_q,
            "q_continuous": expected_q,
            "probs": probs.unsqueeze(-1),
            "logits": logits,
        }

    def forward(
        self,
        start_visual_groups: Sequence[Sequence[torch.Tensor]],
        start_classification_groups: Sequence[torch.Tensor],
        start_relative_time_groups: Sequence[torch.Tensor],
        start_absolute_time_groups: Sequence[torch.Tensor],
        end_visual_groups: Sequence[Sequence[torch.Tensor]],
        end_classification_groups: Sequence[torch.Tensor],
        end_relative_time_groups: Sequence[torch.Tensor],
        end_absolute_time_groups: Sequence[torch.Tensor],
    ) -> TimeRefineHeadOutput:
        if len(start_visual_groups) != len(end_visual_groups):
            raise ValueError("Start and end refinement groups must have equal batch size.")
        expected_batch = len(start_visual_groups)
        metadata_groups = (
            start_classification_groups,
            start_relative_time_groups,
            start_absolute_time_groups,
            end_classification_groups,
            end_relative_time_groups,
            end_absolute_time_groups,
        )
        if any(len(groups) != expected_batch for groups in metadata_groups):
            raise ValueError(
                "All refinement metadata groups must match the visual group batch size."
            )
        for branch_name, visual_groups, branch_metadata in (
            (
                "start",
                start_visual_groups,
                (
                    start_classification_groups,
                    start_relative_time_groups,
                    start_absolute_time_groups,
                ),
            ),
            (
                "end",
                end_visual_groups,
                (
                    end_classification_groups,
                    end_relative_time_groups,
                    end_absolute_time_groups,
                ),
            ),
        ):
            for index, (visual_group, *metadata_group) in enumerate(
                zip(visual_groups, *branch_metadata)
            ):
                if any(len(group) != len(visual_group) for group in metadata_group):
                    raise ValueError(
                        f"{branch_name} metadata length must match visual group length "
                        f"for sample {index}."
                    )
        start = self._forward_branch(
            start_visual_groups,
            start_classification_groups,
            start_relative_time_groups,
            start_absolute_time_groups,
            branch_id=0,
            scorer=self.start_scorer,
        )
        end = self._forward_branch(
            end_visual_groups,
            end_classification_groups,
            end_relative_time_groups,
            end_absolute_time_groups,
            branch_id=1,
            scorer=self.end_scorer,
        )
        return TimeRefineHeadOutput(
            pred_start_q=start["q"],
            pred_end_q=end["q"],
            pred_start_q_continuous=start["q_continuous"],
            pred_end_q_continuous=end["q_continuous"],
            start_probs=start["probs"],
            end_probs=end["probs"],
            start_logits=start["logits"],
            end_logits=end["logits"],
        )
