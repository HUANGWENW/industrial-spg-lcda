from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class FrozenOpenCLIPTextEncoder(nn.Module):
    """Frozen OpenCLIP text tower used to build a small prompt cache."""

    def __init__(
        self,
        architecture: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | torch.device = "cpu",
        normalize: bool = True,
    ) -> None:
        super().__init__()
        try:
            import open_clip
        except ImportError as error:
            raise RuntimeError("Install open-clip-torch before using the text encoder") from error

        self.clip = open_clip.create_model(architecture, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(architecture)
        self.normalize = normalize
        self.device = torch.device(device)
        self.clip.to(self.device).eval()
        self.clip.requires_grad_(False)

    def train(self, mode: bool = True) -> FrozenOpenCLIPTextEncoder:
        super().train(False)
        self.clip.eval()
        return self

    @torch.no_grad()
    def forward(self, prompts: Sequence[str]) -> Tensor:
        tokens = self.tokenizer(list(prompts)).to(self.device)
        embeddings = self.clip.encode_text(tokens).float()
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings


@torch.no_grad()
def build_prompt_cache(
    encoder: FrozenOpenCLIPTextEncoder,
    prompts: Sequence[str],
    output_device: str | torch.device = "cpu",
) -> dict[str, Tensor]:
    unique_prompts = list(dict.fromkeys(prompts))
    embeddings = encoder(unique_prompts).to(output_device)
    return {prompt: embedding for prompt, embedding in zip(unique_prompts, embeddings)}
