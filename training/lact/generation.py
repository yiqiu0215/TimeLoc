from __future__ import annotations

from typing import Optional

import torch

from .qwen25_lact import LaCTCache


def _select_next_token(model, outputs) -> torch.Tensor:
    if hasattr(outputs, "logits") and outputs.logits is not None:
        logits = outputs.logits[:, -1, :]
    else:
        logits = model.lm_head(outputs.last_hidden_state[:, -1, :])
    return logits.argmax(dim=-1, keepdim=True)


@torch.no_grad()
def generate_with_timelens_lact(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    second_per_grid_ts=None,
    max_new_tokens: int = 512,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    **kwargs,
) -> torch.Tensor:
    if input_ids.shape[0] != 1:
        raise RuntimeError("LaCT generation currently supports batch_size=1 only.")
    if attention_mask is not None and not bool(attention_mask.all()):
        raise RuntimeError(
            "LaCT generation currently expects an unpadded batch_size=1 prompt."
        )

    device = input_ids.device
    lact_cache = LaCTCache()
    prompt_len = input_ids.shape[1]
    cache_position = torch.arange(prompt_len, device=device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=None,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        cache_position=cache_position,
        use_cache=True,
        return_dict=True,
        lact_cache=lact_cache,
        **kwargs,
    )
    next_token = _select_next_token(model, outputs)
    generated = [next_token]
    past_key_values = outputs.past_key_values

    seq_len = prompt_len + 1
    for _ in range(max_new_tokens - 1):
        if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
            break
        cache_position = torch.tensor([seq_len - 1], device=device)
        outputs = model(
            input_ids=next_token,
            attention_mask=None,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
            lact_cache=lact_cache,
        )
        past_key_values = outputs.past_key_values
        next_token = _select_next_token(model, outputs)
        generated.append(next_token)
        seq_len += 1

    if not generated:
        return input_ids
    return torch.cat([input_ids, *generated], dim=1)
