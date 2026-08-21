"""Detector, visual-language encoder, and adapter modules."""
from spg_lcda.models.film import FeatureFiLM, P5FiLMHook, infer_feature_channels, install_p5_film
from spg_lcda.models.shift import ShiftAlignedOBBModel, ShiftAlignmentHead
from spg_lcda.models.text_encoder import FrozenOpenCLIPTextEncoder, build_prompt_cache

__all__ = [
    "FeatureFiLM",
    "FrozenOpenCLIPTextEncoder",
    "P5FiLMHook",
    "ShiftAlignedOBBModel",
    "ShiftAlignmentHead",
    "build_prompt_cache",
    "infer_feature_channels",
    "install_p5_film",
]
