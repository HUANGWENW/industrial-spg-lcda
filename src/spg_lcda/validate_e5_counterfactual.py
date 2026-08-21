from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from ultralytics.models.yolo.obb import OBBValidator

from spg_lcda.config import load_config
from spg_lcda.models import FrozenOpenCLIPTextEncoder, build_prompt_cache
from spg_lcda.prompts import counterfactual_prompt_for


CONDITIONS = ("correct", "fixed", "shuffled")


def normalized_path(path: str | Path) -> str:
    return Path(path).resolve(strict=False).as_posix()


def manifest_prompts(
    data_root: Path,
    manifest_path: Path,
    condition: str,
    seed: int,
) -> dict[str, str]:
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        return {
            normalized_path(data_root / row["image_path"]): counterfactual_prompt_for(
                row["transform_name"], condition, seed
            )
            for row in csv.DictReader(handle)
        }


class CounterfactualOBBValidator(OBBValidator):
    def __init__(
        self,
        prompt_by_image: dict[str, str],
        prompt_cache: dict[str, torch.Tensor],
        **kwargs,
    ):
        self.prompt_by_image = prompt_by_image
        self.prompt_cache = prompt_cache
        super().__init__(**kwargs)

    def init_metrics(self, model) -> None:
        self.inference_model = model
        super().init_metrics(model)

    def preprocess(self, batch: dict) -> dict:
        batch = super().preprocess(batch)
        embeddings = torch.stack(
            [
                self.prompt_cache[self.prompt_by_image[normalized_path(path)]]
                for path in batch["im_file"]
            ]
        ).to(self.device)
        detector = getattr(self.inference_model, "model", self.inference_model)
        detector.prompt_film.set_evaluation_text_features(embeddings)
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate E5-S with counterfactual prompts")
    parser.add_argument("--config", default="configs/experiment/e5_s.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data = config["data"]
    model = config["model"]
    text = model["text_encoder"]
    data_root = Path(data["root"])
    manifest_path = data_root / data["val_manifest"]

    prompt_maps = {
        condition: manifest_prompts(data_root, manifest_path, condition, args.seed)
        for condition in CONDITIONS
    }
    encoder = FrozenOpenCLIPTextEncoder(
        architecture=text["architecture"],
        pretrained=text["pretrained"],
        device="cpu",
        normalize=text["normalize"],
    )
    prompt_cache = build_prompt_cache(
        encoder,
        [prompt for mapping in prompt_maps.values() for prompt in mapping.values()],
        output_device="cpu",
    )
    del encoder

    run_id = Path(args.weights).parent.parent.name
    output_dir = Path(args.output or config["output"]["project"]) / "counterfactual" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for condition in CONDITIONS:
        validator = CounterfactualOBBValidator(
            prompt_by_image=prompt_maps[condition],
            prompt_cache=prompt_cache,
            args={
                "model": args.weights,
                "data": data["yolo_yaml"],
                "imgsz": model["image_size"],
                "batch": args.batch_size,
                "workers": args.workers,
                "device": args.device,
                "task": "obb",
                "split": "val",
                "single_cls": True,
                "project": str(output_dir),
                "name": condition,
                "exist_ok": True,
                "plots": True,
            },
        )
        metrics = validator(model=args.weights)
        rows.append(
            {"condition": condition, **{key: float(value) for key, value in metrics.items()}}
        )

    summary_path = output_dir / "counterfactual_metrics.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Counterfactual summary: {summary_path}")


if __name__ == "__main__":
    main()
