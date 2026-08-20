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
