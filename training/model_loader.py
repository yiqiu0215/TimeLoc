"""Model/config/processor loader for TimeLens Qwen3 and Qwen2.5-VL variants."""

from pathlib import Path

from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor


def _validate_model_path(model_path: str) -> None:
    model_path_lower = model_path.lower()
    supported = (
        "qwen3" in model_path_lower
        or "timelens-2b" in model_path_lower
        or "timelens-8b" in model_path_lower
        or "qwen2.5-vl" in model_path_lower
        or "qwen2.5_vl" in model_path_lower
        or "timelens-3b" in model_path_lower
        or "timelens-7b" in model_path_lower
    )
    is_local_checkpoint = (Path(model_path) / "config.json").is_file()
    if not supported and not is_local_checkpoint:
        raise ValueError(
            f"Unsupported model_path={model_path!r}. "
            "Expected Qwen3-VL/TimeLens-2B/8B or Qwen2.5-VL/TimeLens-3B/7B."
        )


def get_model_class(model_path: str, use_residual_tokens: bool = False):
    if use_residual_tokens:
        from training.models import RITQwen3VLForConditionalGeneration

        return RITQwen3VLForConditionalGeneration
    _validate_model_path(model_path)
    return AutoModelForImageTextToText


def get_config_class(model_path: str):
    _validate_model_path(model_path)
    return AutoConfig


def get_processor_class(model_path: str):
    _validate_model_path(model_path)
    return AutoProcessor
