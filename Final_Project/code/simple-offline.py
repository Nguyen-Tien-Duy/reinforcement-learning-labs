from __future__ import annotations

import argparse
from pathlib import Path

from utils.load_data import (
    TransitionValidationError,
    build_d3rlpy_dataset,
    format_validation_report,
    load_transition_dataframe,
    validate_transition_dataframe,
)

BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline RL parquet loader + transition schema validator"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=BASE_DIR.parent / "Data" / "data_2024-04-10_2026-04-10.parquet",
        help="Path to parquet file",
    )
    parser.add_argument(
        "--mode",
        choices=("strict", "lenient"),
        default="strict",
        help="Validation mode",
    )
    parser.add_argument(
        "--to-d3rlpy",
        action="store_true",
        help="Build a d3rlpy dataset object after validation passes",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"Input parquet not found: {args.input}")
        return 1

    dataframe = load_transition_dataframe(args.input)
    result = validate_transition_dataframe(dataframe, mode=args.mode)
    print(format_validation_report(result))

    if not result.passed:
        print("Validation failed. Training is blocked until required transition fields are fixed.")
        return 2

    if not args.to_d3rlpy:
        print("Validation passed. Skip d3rlpy conversion because --to-d3rlpy was not requested.")
        return 0

    try:
        dataset = build_d3rlpy_dataset(dataframe, mode=args.mode)
    except TransitionValidationError as exc:
        print(f"d3rlpy conversion blocked by validation errors: {exc}")
        return 3
    except RuntimeError as exc:
        print(f"d3rlpy runtime dependency error: {exc}")
        return 3
    except ValueError as exc:
        print(f"d3rlpy conversion value error: {exc}")
        return 3

    print(
        "d3rlpy dataset created successfully "
        f"with {len(dataframe)} transitions and type {type(dataset).__name__}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
