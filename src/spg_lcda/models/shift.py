from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from ultralytics.nn.tasks import OBBModel


class ShiftAlignmentHead(nn.Module):
    """Align paired P5 changes with paired CLIP text changes."""

    def __init__(self, channels: int, text_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(channels, text_dim, bias=False)
        self._text_features: Tensor | None = None
        self._pair_indices: Tensor | None = None

    def set_batch(self, text_features: Tensor, pair_indices: Tensor) -> None:
        self._text_features = text_features
        self._pair_indices = pair_indices

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        visual = self.projection(features.mean(dim=(2, 3)))
        pairs = self._pair_indices
        text = self._text_features
        visual_shift = visual[pairs[:, 1]] - visual[pairs[:, 0]]
        text_shift = text[pairs[:, 1]] - text[pairs[:, 0]]
        cosine = F.cosine_similarity(visual_shift, text_shift, dim=-1)
        return 1.0 - cosine.mean(), cosine.mean(), visual.new_tensor(len(pairs))


class ShiftAlignedOBBModel(OBBModel):
    """YOLO11-OBB model with an auxiliary paired domain-shift loss."""

    def configure_shift(self, channels: int, text_dim: int, weight: float) -> None:
        parameter = next(self.parameters())
        self.prompt_shift = ShiftAlignmentHead(channels, text_dim).to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        self.shift_weight = weight
        self.prompt_film.capture_training_features()
        self.shift_audit: dict[str, float] = {}

    def loss(self, batch: dict, preds=None):
        detection_loss, detection_items = super().loss(batch, preds)
        if self.training:
            features = self.prompt_film.take_captured_features()
            shift_loss, shift_cosine, pair_count = self.prompt_shift(features)
            total_loss = detection_loss + self.shift_weight * shift_loss * len(batch["img"])
            self.shift_audit = {
                "shift_loss": float(shift_loss.detach()),
                "shift_cosine": float(shift_cosine.detach()),
                "pair_count": float(pair_count.detach()),
            }
        else:
            total_loss = detection_loss
            shift_loss = detection_loss.new_zeros(())
            shift_cosine = detection_loss.new_zeros(())
            pair_count = detection_loss.new_zeros(())

        shift_items = torch.stack(
            (shift_loss.detach(), shift_cosine.detach(), pair_count.detach())
        )
        return total_loss, torch.cat((detection_items, shift_items))
