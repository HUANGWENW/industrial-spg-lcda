from argparse import Namespace

from spg_lcda.config import load_config
from spg_lcda.prompts import (
    GENERIC_PROMPT,
    PHOTOMETRIC_PROMPTS,
    counterfactual_prompt_for,
    prompt_for,
)
from spg_lcda.train_e5 import ultralytics_overrides


def test_e5_configs_change_only_prompt_policy_and_output_name() -> None:
    generic = load_config("configs/experiment/e5_g.yaml")
    structured = load_config("configs/experiment/e5_s.yaml")

    assert generic["model"] == structured["model"]
    assert generic["data"] == structured["data"]
    assert generic["training"] == structured["training"]
    assert generic["loss"] == structured["loss"]
    assert generic["prompt"]["mode"] == "generic"
    assert structured["prompt"]["mode"] == "manifest"
    assert generic["model"]["detector"] == "yolo11n-obb.pt"
    assert generic["model"]["image_size"] == 1280
    assert generic["model"]["film"]["module_index"] == 10
    assert generic["training"]["mosaic"] == 0.0
    assert generic["training"]["hsv_v"] == 0.0


def test_photometric_prompts_do_not_claim_geometry() -> None:
    forbidden = ("single", "angle", "rotation", "position", "left", "right")
    for prompt in [GENERIC_PROMPT, *PHOTOMETRIC_PROMPTS.values()]:
        assert not any(token in prompt.lower() for token in forbidden)


def test_prompt_policy() -> None:
    assert prompt_for("brightness_low", "generic") == GENERIC_PROMPT
    assert prompt_for("brightness_low", "manifest") == PHOTOMETRIC_PROMPTS["brightness_low"]


def test_shuffled_counterfactual_is_balanced_and_always_wrong() -> None:
    names = tuple(PHOTOMETRIC_PROMPTS)
    shuffled = [counterfactual_prompt_for(name, "shuffled", seed=42) for name in names]
    correct = [PHOTOMETRIC_PROMPTS[name] for name in names]

    assert set(shuffled) == set(correct)
    assert all(left != right for left, right in zip(shuffled, correct))


def test_e5_training_overrides_use_1280_and_seeded_output() -> None:
    config = load_config("configs/experiment/e5_g.yaml")
    args = Namespace(
        epochs=1,
        batch_size=2,
        workers=0,
        seed=43,
        fraction=0.03,
        device="0",
        name=None,
    )
    overrides = ultralytics_overrides(config, args)
    assert overrides["imgsz"] == 1280
    assert overrides["name"] == "E5-G_seed43"
    assert overrides["mosaic"] == 0.0
    assert overrides["fraction"] == 0.03


def test_e5_f_inherits_all_e5_s_conditions() -> None:
    structured = load_config("configs/experiment/e5_s.yaml")
    shift = load_config("configs/experiment/e5_f.yaml")

    for key in ("model", "data", "training", "prompt"):
        assert shift[key] == structured[key]
    assert structured["loss"]["shift_weight"] == 0.0
    assert shift["loss"]["shift_weight"] == 0.01
