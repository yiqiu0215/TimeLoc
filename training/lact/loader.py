from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import warnings

import torch
from transformers import AutoConfig, AutoModelForImageTextToText

from .config import LACT_PARAM_KEYWORDS, LaCTConfig, load_lact_config, validate_lact_config
from .qwen25_lact import wrap_qwen25_vl_model_with_lact


def _convert_qwen25_vl_checkpoint_key(key: str) -> str:
    if key.startswith("visual."):
        return f"model.{key}"
    if key.startswith("model."):
        if key.startswith("model.language_model.") or key.startswith("model.visual."):
            return key
        return f"model.language_model.{key[len('model.'):]}"
    return key


def _find_checkpoint_files(checkpoint_path: Path) -> list[Path]:
    index_names = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
    for index_name in index_names:
        index_path = checkpoint_path / index_name
        if not index_path.exists():
            continue
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        files = sorted({checkpoint_path / name for name in weight_map.values()})
        if files:
            return files

    single_file_names = ("model.safetensors", "pytorch_model.bin")
    for file_name in single_file_names:
        file_path = checkpoint_path / file_name
        if file_path.exists():
            return [file_path]

    safetensor_files = sorted(checkpoint_path.glob("model-*.safetensors"))
    if safetensor_files:
        return safetensor_files

    bin_files = sorted(checkpoint_path.glob("pytorch_model-*.bin"))
    if bin_files:
        return bin_files

    raise FileNotFoundError(
        f"Cannot find HF model weights in checkpoint path: {checkpoint_path}"
    )


def _load_checkpoint_file(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")

    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return state_dict


def _format_keys(keys: list[str], limit: int = 40) -> str:
    shown = keys[:limit]
    suffix = "" if len(keys) <= limit else f"\n... and {len(keys) - limit} more"
    return "\n".join(shown) + suffix


def _load_converted_checkpoint(model, checkpoint_path: Path) -> None:
    model_state_keys = set(model.state_dict().keys())
    parameter_keys = set(dict(model.named_parameters()).keys())
    loaded_keys: set[str] = set()
    unexpected_keys: list[str] = []

    for shard_path in _find_checkpoint_files(checkpoint_path):
        shard_state = _load_checkpoint_file(shard_path)
        converted_state = {}
        for key, value in shard_state.items():
            converted_key = _convert_qwen25_vl_checkpoint_key(key)
            if converted_key in model_state_keys:
                converted_state[converted_key] = value
                loaded_keys.add(converted_key)
            else:
                unexpected_keys.append(f"{key} -> {converted_key}")
        model.load_state_dict(converted_state, strict=False)
        del shard_state, converted_state

    missing_parameter_keys = sorted(parameter_keys - loaded_keys)
    if missing_parameter_keys:
        lact_missing = [
            key
            for key in missing_parameter_keys
            if any(keyword in key for keyword in LACT_PARAM_KEYWORDS)
        ]
        details = _format_keys(lact_missing or missing_parameter_keys)
        raise RuntimeError(
            "LaCT checkpoint loading left model parameters missing. "
            "This usually means the checkpoint structure and LaCT wrapper do not match.\n"
            f"{details}"
        )

    if unexpected_keys:
        warnings.warn(
            "Skipped checkpoint weights that do not exist in the wrapped LaCT model:\n"
            f"{_format_keys(sorted(unexpected_keys))}",
            RuntimeWarning,
        )


def _check_no_meta_parameters(model) -> None:
    meta_parameters = [
        name for name, param in model.named_parameters() if getattr(param, "is_meta", False)
    ]
    if meta_parameters:
        raise RuntimeError(
            "LaCT checkpoint loading left meta tensors in the model:\n"
            f"{_format_keys(meta_parameters)}"
        )


def _resolve_eval_device(device_map) -> torch.device:
    if isinstance(device_map, str) and device_map == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_qwen25_lact_model(
    model_path: str,
    lact_config: LaCTConfig,
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "flash_attention_2",
    device_map: str = "auto",
    trust_remote_code: bool = True,
):
    if not lact_config.lact_enable:
        raise ValueError("load_qwen25_lact_model requires lact_enable=True.")
    if load_lact_config(model_path) is None:
        warnings.warn(
            "lact_config.json was not found; using the explicit LaCT arguments passed "
            "to load_qwen25_lact_model. Prefer evaluating LaCT checkpoints with the "
            "saved lact_config.json.",
            RuntimeWarning,
        )
    else:
        validate_lact_config(model_path, lact_config)

    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    )
    model_type = getattr(config, "model_type", "")
    if "qwen2_5_vl" not in model_type and "qwen2.5" not in str(config).lower():
        raise ValueError(
            f"LaCT loader currently supports Qwen2.5-VL checkpoints only, got model_type={model_type!r}."
        )

    checkpoint_path = Path(model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {model_path}")

    model = AutoModelForImageTextToText.from_config(
        config,
        trust_remote_code=trust_remote_code,
    )
    model = wrap_qwen25_vl_model_with_lact(model, lact_config)
    model.to(dtype=torch_dtype)

    _load_converted_checkpoint(model, checkpoint_path)
    _check_no_meta_parameters(model)

    model.to(_resolve_eval_device(device_map))
    model.eval()
    return model
