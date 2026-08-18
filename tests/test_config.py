from spg_lcda.config import load_config


def test_smoke_config_loads() -> None:
    config = load_config("configs/experiment/smoke.yaml")
    assert config["model"]["num_classes"] == 1
    assert config["experiment"]["seed"] == 42


def test_full_config_inherits_model_and_overrides_losses() -> None:
    config = load_config("configs/experiment/e7_full.yaml")
    assert config["model"]["detector"] == "yolov8n"
    assert config["loss"]["shift_weight"] == 0.1
    assert config["loss"]["hard_negative_weight"] == 0.1
    assert config["loss"]["localization_weight"] == 0.1
