from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FeatureFiLM(nn.Module):
    """Generate channel-wise FiLM parameters from one text embedding per image."""

    def __init__(
        self,
        text_dim: int,
        channels: int,
        normalize_text: bool = True,
        identity_initialization: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.normalize_text = normalize_text
        self.projection = nn.Linear(text_dim, channels * 2)
        if identity_initialization:
            nn.init.zeros_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)

    def forward(self, features: Tensor, text_features: Tensor) -> Tensor:
        if features.ndim != 4:
            raise ValueError(f"FiLM expects BCHW features, got shape {tuple(features.shape)}")
        if len(features) != len(text_features):
            raise ValueError("Feature and text batch sizes must match")
        if self.normalize_text:
            text_features = F.normalize(text_features, dim=-1)
        gamma, beta = self.projection(text_features).chunk(2, dim=-1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return features * (1.0 + gamma) + beta


class P5FiLMHook(nn.Module):
    """Trainable forward hook for an Ultralytics backbone P5 module."""

    def __init__(self, film: FeatureFiLM) -> None:
        super().__init__()
        self.film = film
        self._text_features: Tensor | None = None

    def set_text_features(self, text_features: Tensor) -> None:
        self._text_features = text_features

    def clear_text_features(self) -> None:
        self._text_features = None

    def apply(self, _module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> Tensor:
        if self._text_features is None:
            raise RuntimeError("Set text features before the detector forward pass")
        text_features = self._text_features.to(device=output.device, dtype=output.dtype)
        return self.film(output, text_features)


def _ultralytics_layers(model: nn.Module) -> nn.ModuleList:
    layers = getattr(model, "model", None)
    if isinstance(layers, nn.ModuleList):
        return layers
    inner = getattr(layers, "model", None)
    if isinstance(inner, nn.ModuleList):
        return inner
    raise TypeError("Expected an Ultralytics YOLO wrapper or OBBModel")


@torch.no_grad()
def infer_feature_channels(
    model: nn.Module,
    module_index: int,
    image_size: int,
    device: str | torch.device,
) -> int:
    captured: list[int] = []

    def capture(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        captured.append(output.shape[1])

    layers = _ultralytics_layers(model)
    handle = layers[module_index].register_forward_hook(capture)
    was_training = model.training
    model.eval()
    try:
        model(torch.zeros(1, 3, image_size, image_size, device=device))
    finally:
        handle.remove()
        model.train(was_training)
    if not captured:
        raise RuntimeError(f"YOLO module {module_index} did not produce a tensor feature")
    return captured[-1]


def install_p5_film(
    model: nn.Module,
    text_dim: int,
    channels: int,
    module_index: int = 10,
    normalize_text: bool = True,
    identity_initialization: bool = True,
) -> tuple[P5FiLMHook, Callable[[], None]]:
    """Attach FiLM and register its parameters on the detector model."""
    if hasattr(model, "prompt_film"):
        raise ValueError("The model already has a prompt_film adapter")
    model_parameter = next(model.parameters())
    hook = P5FiLMHook(
        FeatureFiLM(
            text_dim,
            channels,
            normalize_text=normalize_text,
            identity_initialization=identity_initialization,
        )
    ).to(device=model_parameter.device, dtype=model_parameter.dtype)
    model.add_module("prompt_film", hook)
    layers = _ultralytics_layers(model)
    handle = layers[module_index].register_forward_hook(hook.apply)
    return hook, handle.remove
