import argparse
import json

from spg_lcda.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="SPG-LCDA experiment entry point")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return
    raise SystemExit("Training implementation is not added yet; run with --dry-run.")


if __name__ == "__main__":
    main()

