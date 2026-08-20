from __future__ import annotations

import csv
from pathlib import Path

import torch
from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import de_parallel

from spg_lcda.models import (
    FrozenOpenCLIPTextEncoder,
    build_prompt_cache,
    infer_feature_channels,
    install_p5_film,
)
from spg_lcda.prompts import prompt_for


def _normalized_path(path: str | Path) -> str:
    return Path(path).resolve(strict=False).as_posix()


def _manifest_prompt_map(
    data_root: Path,
    manifest_paths: list[Path],
    prompt_mode: str,
) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for manifest_path in manifest_paths:
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                image_path = _normalized_path(data_root / row["image_path"])
                prompts[image_path] = prompt_for(row["transform_name"], prompt_mode)
    return prompts


class PromptConditionedOBBTrainer(OBBTrainer):
    """Ultralytics OBB trainer that conditions backbone P5 on cached prompt embeddings."""

    def __init__(self, experiment_config: dict, overrides: dict) -> None:
        self.experiment_config = experiment_config
        data_config = experiment_config["data"]
        data_root = Path(data_config["root"])
        manifests = [
            data_root / data_config["train_manifest"],
            data_root / data_config["val_manifest"],
        ]
        self.prompt_by_image = _manifest_prompt_map(
            data_root,
            manifests,
            experiment_config["prompt"]["mode"],
        )
        self.prompt_cache: dict[str, torch.Tensor] = {}
        super().__init__(overrides=overrides)

    def _build_prompt_cache(self) -> None:
        model_config = self.experiment_config["model"]
        text_config = model_config["text_encoder"]
        evaluation_prompt = self.experiment_config["prompt"]["evaluation_text"]
        prompts = [*self.prompt_by_image.values(), evaluation_prompt]
        encoder = FrozenOpenCLIPTextEncoder(
            architecture=text_config["architecture"],
            pretrained=text_config["pretrained"],
            device="cpu",
            normalize=text_config["normalize"],
        )
        self.prompt_cache = build_prompt_cache(encoder, prompts, output_device="cpu")
        del encoder
        actual_dim = next(iter(self.prompt_cache.values())).shape[0]
        if actual_dim != text_config["embedding_dim"]:
            raise RuntimeError(
                f"Configured text dimension {text_config['embedding_dim']} "
                f"does not match OpenCLIP output {actual_dim}"
            )

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = OBBModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        model_config = self.experiment_config["model"]
        film_config = model_config["film"]
        text_config = model_config["text_encoder"]
        channels = infer_feature_channels(
            model,
            module_index=film_config["module_index"],
            image_size=film_config["probe_image_size"],
            device=next(model.parameters()).device,
        )
        film, _remove_hook = install_p5_film(
            model,
            text_dim=text_config["embedding_dim"],
            channels=channels,
            module_index=film_config["module_index"],
            normalize_text=film_config["normalize_text"],
            identity_initialization=film_config["identity_initialization"],
        )
        self._build_prompt_cache()
        if weights:
            model.load(weights)
        evaluation_prompt = self.experiment_config["prompt"]["evaluation_text"]
        film.set_evaluation_text_feature(self.prompt_cache[evaluation_prompt])
        return model

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        embeddings = []
        for image_path in batch["im_file"]:
            key = _normalized_path(image_path)
            try:
                prompt = self.prompt_by_image[key]
            except KeyError as error:
                raise KeyError(f"Image is missing from E5 manifests: {image_path}") from error
            embeddings.append(self.prompt_cache[prompt])
        model = de_parallel(self.model)
        model.prompt_film.set_text_features(torch.stack(embeddings).to(self.device))
        return batch
