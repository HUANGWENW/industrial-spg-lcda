#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch
from ultralytics import YOLO

from spg_lcda.config import load_config
from spg_lcda.models import (
    FrozenOpenCLIPTextEncoder,
    infer_feature_channels,
    install_p5_film,
)
from spg_lcda.prompts import GENERIC_PROMPT, PHOTOMETRIC_PROMPTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Check YOLO11-OBB + OpenCLIP + P5 FiLM")
    parser.add_argument("--config", default="configs/experiment/e5_g.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(f"cuda:{config['training']['device']}")
    model_config = config["model"]
    text_config = model_config["text_encoder"]
    film_config = model_config["film"]

    detector = YOLO(model_config["detector"]).model.to(device)
    encoder = FrozenOpenCLIPTextEncoder(
        architecture=text_config["architecture"],
        pretrained=text_config["pretrained"],
        device=device,
        normalize=text_config["normalize"],
    )
    prompts = [GENERIC_PROMPT, *PHOTOMETRIC_PROMPTS.values()]
    text_features = encoder(prompts)
    if text_features.shape[1] != text_config["embedding_dim"]:
        raise RuntimeError(
            f"Configured text dimension {text_config['embedding_dim']} does not match "
            f"OpenCLIP output {text_features.shape[1]}"
        )

    channels = infer_feature_channels(
        detector,
        module_index=film_config["module_index"],
        image_size=model_config["image_size"],
        device=device,
    )
    film, remove_hook = install_p5_film(
        detector,
        text_dim=text_features.shape[1],
        channels=channels,
        module_index=film_config["module_index"],
        normalize_text=film_config["normalize_text"],
        identity_initialization=film_config["identity_initialization"],
    )
    film.set_text_features(text_features[:2])
    detector.eval()
    with torch.no_grad():
        dummy_images = torch.zeros(
            2,
            3,
            model_config["image_size"],
            model_config["image_size"],
            device=device,
        )
        detector(dummy_images)
    remove_hook()

    print(f"detector={model_config['detector']}")
    print(f"text_embeddings={tuple(text_features.shape)}")
    print(f"film_module_index={film_config['module_index']}")
    print(f"film_channels={channels}")
    print("E5 common model check passed")


if __name__ == "__main__":
    main()
