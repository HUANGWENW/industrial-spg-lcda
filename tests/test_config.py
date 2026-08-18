from spg_lcda.config import load_config


def test_smoke_config_loads() -> None:
    config = load_config("configs/experiment/smoke.yaml")
    assert config["model"]["num_classes"] == 1
    assert config["experiment"]["seed"] == 42

