GENERIC_PROMPT = "an industrial part image under unspecified lighting conditions"

PHOTOMETRIC_PROMPTS = {
    "identity": (
        "an industrial part image under normal brightness, normal contrast, "
        "and neutral color temperature"
    ),
    "brightness_low": (
        "an industrial part image under low brightness, normal contrast, "
        "and neutral color temperature"
    ),
    "contrast_low": (
        "an industrial part image under normal brightness, low contrast, "
        "and neutral color temperature"
    ),
    "temperature_warm": (
        "an industrial part image under normal brightness, normal contrast, "
        "and warm color temperature"
    ),
}


def prompt_for(transform_name: str, mode: str) -> str:
    if mode == "generic":
        return GENERIC_PROMPT
    if mode == "manifest":
        try:
            return PHOTOMETRIC_PROMPTS[transform_name]
        except KeyError as error:
            raise ValueError(f"Unknown photometric transform: {transform_name}") from error
    raise ValueError(f"Unknown prompt mode: {mode}")


def counterfactual_prompt_for(transform_name: str, condition: str, seed: int = 42) -> str:
    """Return the correct, fixed, or deterministically permuted prompt."""
    if condition == "correct":
        return PHOTOMETRIC_PROMPTS[transform_name]
    if condition == "fixed":
        return PHOTOMETRIC_PROMPTS["identity"]
    if condition == "shuffled":
        names = tuple(PHOTOMETRIC_PROMPTS)
        shift = seed % (len(names) - 1) + 1
        index = names.index(transform_name)
        return PHOTOMETRIC_PROMPTS[names[(index + shift) % len(names)]]
    raise ValueError(f"Unknown counterfactual condition: {condition}")
