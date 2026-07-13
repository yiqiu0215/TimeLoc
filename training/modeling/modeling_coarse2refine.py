"""Canonical Coarse2Refine model import path.

The implementation remains in the historical module so existing checkpoint
imports continue to work; the exported class and serialized architecture name
are canonical Coarse2Refine names.
"""

from training.modeling.modeling_timelens_refine import (
    Coarse2RefineForConditionalGeneration,
    TimeLensRefineForConditionalGeneration,
)

__all__ = [
    "Coarse2RefineForConditionalGeneration",
    "TimeLensRefineForConditionalGeneration",
]
