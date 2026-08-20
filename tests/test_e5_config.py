from spg_lcda.config import load_config
from spg_lcda.prompts import GENERIC_PROMPT, PHOTOMETRIC_PROMPTS, prompt_for


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
