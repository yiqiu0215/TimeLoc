from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from transformers.utils import ModelOutput


@dataclass
class Coarse2RefineOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    ntp_loss: Optional[torch.Tensor] = None
    diou_loss: Optional[torch.Tensor] = None
    smooth_l1_loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    pred_start: Optional[torch.Tensor] = None
    pred_end: Optional[torch.Tensor] = None
    pred_start_q: Optional[torch.Tensor] = None
    pred_end_q: Optional[torch.Tensor] = None
    start_probs: Optional[torch.Tensor] = None
    end_probs: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    past_key_values: Optional[Tuple[torch.Tensor, ...]] = None
    window_edge_cases: Optional[torch.Tensor] = None


@dataclass
class Coarse2RefineInferenceOutput(ModelOutput):
    pred_start: Optional[torch.Tensor] = None
    pred_end: Optional[torch.Tensor] = None
    generated_ids: Optional[torch.Tensor] = None
    prompt_length: Optional[int] = None
    statuses: Optional[Tuple[str, ...]] = None
    parse_statuses: Optional[Tuple[str, ...]] = None
    coarse_labels: Optional[Tuple[Tuple[int, ...], ...]] = None
    coarse_time_bins: Optional[Tuple[Tuple[int, ...], ...]] = None
    candidate_windows: Optional[Tuple[Any, ...]] = None


# Read-only compatibility aliases for code written against the previous names.
TimeLensRefineOutput = Coarse2RefineOutput
TimeLensRefineInferenceOutput = Coarse2RefineInferenceOutput
