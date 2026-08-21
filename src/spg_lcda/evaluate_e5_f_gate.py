from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from spg_lcda.config import load_config


MAP_KEY = "metrics/mAP50-95(B)"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the E5-F continuation gates")
    parser.add_argument("--config", default="configs/experiment/e5_f.yaml")
    parser.add_argument("--audit", required=True)
    parser.add_argument("--counterfactual", required=True)
    parser.add_argument("--baseline-map5095", required=True, type=float)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    criteria = load_config(args.config)["continuation"]
    audit = read_rows(Path(args.audit))
    counterfactual = {
        row["condition"]: float(row[MAP_KEY])
        for row in read_rows(Path(args.counterfactual))
    }

    window = min(3, len(audit))
    first_cosine = sum(float(row["shift_cosine"]) for row in audit[:window]) / window
    last_cosine = sum(float(row["shift_cosine"]) for row in audit[-window:]) / window
    cosine_gain = last_cosine - first_cosine
    correct_gap = counterfactual["correct"] - max(
        counterfactual["fixed"], counterfactual["shuffled"]
    )
    synthetic_drop = args.baseline_map5095 - counterfactual["correct"]
    gradient_active = max(float(row["film_grad_norm"]) for row in audit) > 0 and max(
        float(row["shift_grad_norm"]) for row in audit
    ) > 0

    gates = {
        "gradient_active": gradient_active,
        "shift_cosine_gain": cosine_gain >= criteria["min_shift_cosine_gain"],
        "correct_prompt_gap": correct_gap >= criteria["min_counterfactual_map5095_gap"],
        "synthetic_retention": synthetic_drop <= criteria["max_synthetic_map5095_drop"],
    }
    report = {
        "decision": (
            "continue_to_real_validation" if all(gates.values()) else "stop_and_redesign_data"
        ),
        "gates": gates,
        "observed": {
            "shift_cosine_gain": cosine_gain,
            "correct_prompt_map5095_gap": correct_gap,
            "synthetic_map5095_drop": synthetic_drop,
        },
        "real_validation_gate": {
            "required_map5095_gain": criteria["min_real_map5095_gain"],
            "status": "pending_manual_real_domain_evaluation",
        },
    }
    output = Path(args.output or Path(args.audit).with_name("continuation_decision.json"))
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Continuation decision: {output}")


if __name__ == "__main__":
    main()
