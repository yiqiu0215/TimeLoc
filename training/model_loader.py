"""Model/config/processor loader for Qwen3, TimeLens, and Coarse2Refine."""

import json
from pathlib import Path
from typing import Optional

from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from training.modeling.special_tokens import is_qwen25_coarse2refine


def _validate_model_path(model_path: str) -> None:
    model_path_lower = model_path.lower()
    supported = (
        "qwen3" in model_path_lower
        or "timelens-2b" in model_path_lower
        or "timelens-8b" in model_path_lower
        or "qwen2.5-vl" in model_path_lower
        or "qwen2.5_vl" in model_path_lower
        or "qwen25-vl" in model_path_lower
        or "coarse2refine" in model_path_lower
        # Keep legacy TimeLens model paths loadable; this is not a
        # Coarse2Refine trigger.
        or "timelens-3b" in model_path_lower
        or "timelens-7b" in model_path_lower
    )
    if not supported:
        raise ValueError(
            f"Unsupported model_path={model_path!r}. "
            "Expected Qwen3-VL/TimeLens or Qwen2.5-VL/Coarse2Refine paths."
        )


def get_model_class(model_path: str):
    _validate_model_path(model_path)
    return AutoModelForImageTextToText


def get_config_class(model_path: str):
    _validate_model_path(model_path)
    return AutoConfig


def get_processor_class(model_path: str):
    _validate_model_path(model_path)
    return AutoProcessor


def is_coarse2refine_checkpoint(
    model_path: str,
    model_id: Optional[str] = None,
    processor_path: Optional[str] = None,
) -> bool:
    """Authoritatively identify a Coarse2Refine checkpoint.

    Tries the fast name-based heuristic first (covers Hub repo ids that have
    no local ``config.json`` yet), then falls back to reading ``model_type``
    from a local checkpoint directory. Callers (evaluation entry point,
    Dataset, ...) must consume this result instead of re-deriving their own.

    ``processor_path`` is accepted for call-site compatibility, but it is
    deliberately not a trigger.  ``timelens_refine`` remains a read-only
    compatibility value for checkpoints produced before the rename.
    """
    if is_qwen25_coarse2refine(model_id, model_path, processor_path):
        return True
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return False
    with config_path.open("r", encoding="utf-8") as reader:
        config_data = json.load(reader)
    model_type = config_data.get("model_type")
    return model_type in {"coarse2refine", "timelens_refine"}


def is_time_refine_checkpoint(
    model_path: str,
    model_id: Optional[str] = None,
    processor_path: Optional[str] = None,
) -> bool:
    """Deprecated compatibility alias for :func:`is_coarse2refine_checkpoint`."""

    return is_coarse2refine_checkpoint(model_path, model_id, processor_path)
