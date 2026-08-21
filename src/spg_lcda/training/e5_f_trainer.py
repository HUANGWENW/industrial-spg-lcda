from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
from torch.utils.data import DataLoader, Sampler
from ultralytics.data.build import seed_worker
from ultralytics.data.utils import PIN_MEMORY
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first

from spg_lcda.models import ShiftAlignedOBBModel
from spg_lcda.training.e5_trainer import PromptConditionedOBBTrainer, _normalized_path


class SourceGroupedBatchSampler(Sampler[list[int]]):
    """Keep all four photometric variants of each source in the same batch."""

    def __init__(self, groups: list[list[int]], batch_size: int, seed: int) -> None:
        self.groups = groups
        self.groups_per_batch = batch_size // 4
        self.seed = seed
        self.epoch = 0
        if batch_size % 4:
            raise ValueError("E5-F batch size must be divisible by four")

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.groups), generator=generator).tolist()
        self.epoch += 1
        for start in range(0, len(order), self.groups_per_batch):
            batch = []
            for group_index in order[start : start + self.groups_per_batch]:
                batch.extend(self.groups[group_index])
            yield batch

    def __len__(self) -> int:
        return math.ceil(len(self.groups) / self.groups_per_batch)


def _manifest_metadata(
    data_root: Path,
    manifest_path: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    source_by_image = {}
    transform_by_image = {}
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_path = _normalized_path(data_root / row["image_path"])
            source_by_image[image_path] = row["source_id"]
            transform_by_image[image_path] = row["transform_name"]
    return source_by_image, transform_by_image


class ShiftAlignedOBBTrainer(PromptConditionedOBBTrainer):
    """E5-S plus paired visual-text domain-shift alignment."""

    model_class = ShiftAlignedOBBModel

    def __init__(self, experiment_config: dict, overrides: dict) -> None:
        data = experiment_config["data"]
        data_root = Path(data["root"])
        self.source_by_image, self.transform_by_image = _manifest_metadata(
            data_root,
            data_root / data["train_manifest"],
        )
        self._batch_audit = []
        self._film_grad_norms = []
        self._shift_grad_norms = []
        super().__init__(experiment_config, overrides)
        self.shift_audit_csv = self.save_dir / "shift_audit.csv"
        protocol = {
            "loss": experiment_config["loss"],
            "continuation": experiment_config["continuation"],
        }
        (self.save_dir / "e5_f_protocol.json").write_text(
            json.dumps(protocol, indent=2),
            encoding="utf-8",
        )
        self.add_callback("on_train_epoch_start", self._start_shift_epoch)
        self.add_callback("on_train_batch_end", self._collect_shift_batch)
        self.add_callback("on_train_epoch_end", self._finish_shift_epoch)

    def configure_conditioned_model(self, model, channels: int) -> None:
        model.configure_shift(
            channels=channels,
            text_dim=self.experiment_config["model"]["text_encoder"]["embedding_dim"],
            weight=self.experiment_config["loss"]["shift_weight"],
        )

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode == "val":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        grouped_indices = defaultdict(list)
        for index, image_path in enumerate(dataset.im_files):
            grouped_indices[self.source_by_image[_normalized_path(image_path)]].append(index)
        groups = [indices for indices in grouped_indices.values() if len(indices) == 4]
        dropped = len(dataset) - 4 * len(groups)
        if dropped:
            LOGGER.warning("E5-F skipped %d images from incomplete source groups", dropped)
        LOGGER.info(
            "E5-F paired loader: %d source groups, %d shift pairs per epoch, lambda=%.3g",
            len(groups),
            3 * len(groups),
            self.experiment_config["loss"]["shift_weight"],
        )
        sampler = SourceGroupedBatchSampler(groups, batch_size, self.args.seed)
        generator = torch.Generator().manual_seed(6148914691236517205 + RANK)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=self.args.workers,
            pin_memory=PIN_MEMORY,
            collate_fn=dataset.collate_fn,
            worker_init_fn=seed_worker,
            generator=generator,
        )

    def condition_batch(self, model, batch: dict, text_features: torch.Tensor) -> None:
        super().condition_batch(model, batch, text_features)
        grouped = defaultdict(dict)
        for index, image_path in enumerate(batch["im_file"]):
            key = _normalized_path(image_path)
            grouped[self.source_by_image[key]][self.transform_by_image[key]] = index
        pairs = []
        for transforms in grouped.values():
            reference = transforms["identity"]
            pairs.extend(
                (reference, index)
                for name, index in transforms.items()
                if name != "identity"
            )
        pair_indices = torch.tensor(pairs, device=self.device, dtype=torch.long)
        model.prompt_shift.set_batch(text_features, pair_indices)

    def get_validator(self):
        validator = super().get_validator()
        self.loss_names = (
            "box_loss",
            "cls_loss",
            "dfl_loss",
            "shift_loss",
            "shift_cos",
            "shift_pairs",
        )
        return validator

    def optimizer_step(self) -> None:
        self.scaler.unscale_(self.optimizer)
        model = de_parallel(self.model)
        self._film_grad_norms.append(self._grad_norm(model.prompt_film.film))
        self._shift_grad_norms.append(self._grad_norm(model.prompt_shift))
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    @staticmethod
    def _grad_norm(module: torch.nn.Module) -> float:
        gradients = [
            parameter.grad.detach().norm()
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        return float(torch.stack(gradients).norm())

    def _start_shift_epoch(self, _trainer) -> None:
        self._batch_audit.clear()
        self._film_grad_norms.clear()
        self._shift_grad_norms.clear()

    def _collect_shift_batch(self, _trainer) -> None:
        self._batch_audit.append(de_parallel(self.model).shift_audit.copy())

    def _finish_shift_epoch(self, _trainer) -> None:
        row = {
            "epoch": self.epoch + 1,
            "shift_weight": self.experiment_config["loss"]["shift_weight"],
            "shift_loss": mean(item["shift_loss"] for item in self._batch_audit),
            "shift_cosine": mean(item["shift_cosine"] for item in self._batch_audit),
            "pairs_per_batch": mean(item["pair_count"] for item in self._batch_audit),
            "film_grad_norm": mean(self._film_grad_norms),
            "shift_grad_norm": mean(self._shift_grad_norms),
        }
        write_header = not self.shift_audit_csv.exists()
        with self.shift_audit_csv.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=row)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        LOGGER.info(
            "E5-F audit: shift_loss=%.4f shift_cos=%.4f pairs=%.1f "
            "film_grad=%.4g shift_grad=%.4g",
            row["shift_loss"],
            row["shift_cosine"],
            row["pairs_per_batch"],
            row["film_grad_norm"],
            row["shift_grad_norm"],
        )
