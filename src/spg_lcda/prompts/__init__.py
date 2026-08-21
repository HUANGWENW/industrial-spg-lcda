"""Structured simulator prompts and industrial hard-negative prompt banks."""
from spg_lcda.prompts.photometric import (
    GENERIC_PROMPT,
    PHOTOMETRIC_PROMPTS,
    counterfactual_prompt_for,
    prompt_for,
)

__all__ = [
    "GENERIC_PROMPT",
    "PHOTOMETRIC_PROMPTS",
    "counterfactual_prompt_for",
    "prompt_for",
]
