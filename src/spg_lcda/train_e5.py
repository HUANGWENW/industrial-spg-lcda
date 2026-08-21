from __future__ import annotations

import argparse

from spg_lcda.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train E5-G or E5-S with YOLO11-OBB and P5 FiLM")
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--device")
    parser.add_argument("--name")
    return parser.parse_args()


def ultralytics_overrides(config: dict, args: argparse.Namespace) -> dict:
    model = config["model"]
    data = config["data"]
    training = config["training"]
    output = config["output"]
    experiment = config["experiment"]
    seed = experiment["seed"] if args.seed is None else args.seed
    overrides = {
        "model": model["detector"],
        "data": data["yolo_yaml"],
        "epochs": args.epochs or training["epochs"],
        "batch": args.batch_size or training["batch_size"],
        "workers": training["workers"] if args.workers is None else args.workers,
        "imgsz": model["image_size"],
        "patience": training["patience"],
        "close_mosaic": training["close_mosaic"],
        "mosaic": training["mosaic"],
        "mixup": training["mixup"],
        "copy_paste": training["copy_paste"],
        "hsv_h": training["hsv_h"],
        "hsv_s": training["hsv_s"],
        "hsv_v": training["hsv_v"],
        "bgr": training["bgr"],
        "amp": training["amp"],
        "device": training["device"] if args.device is None else args.device,
        "deterministic": training["deterministic"],
        "seed": seed,
        "project": output["project"],
        "name": args.name or f"{output['name']}_seed{seed}",
        "task": "obb",
        "single_cls": True,
        "val": True,
    }
    if args.fraction is not None:
        overrides["fraction"] = args.fraction
    return overrides


def main() -> None:
    from spg_lcda.training import PromptConditionedOBBTrainer, ShiftAlignedOBBTrainer

    args = parse_args()
    config = load_config(args.config)
    trainer_class = (
        ShiftAlignedOBBTrainer
        if config["loss"]["shift_weight"] > 0
        else PromptConditionedOBBTrainer
    )
    trainer = trainer_class(
        experiment_config=config,
        overrides=ultralytics_overrides(config, args),
    )
    trainer.train()


if __name__ == "__main__":
    main()
