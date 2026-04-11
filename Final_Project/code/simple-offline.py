from __future__ import annotations

from datetime import datetime
import argparse
from pathlib import Path

from utils.load_data import (
    TransitionBuildConfig,
    TransitionValidationError,
    build_transitions_from_parquet,
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
    parser.add_argument(
        "--build-from-raw",
        type=Path,
        default=None,
        help="Build transitions from raw parquet instead of loading prebuilt transition parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output parquet path for built transitions",
    )
    parser.add_argument(
        "--action-col",
        type=str,
        default=None,
        help="Raw column used to derive action (action=1 when value > threshold)",
    )
    parser.add_argument(
        "--queue-col",
        type=str,
        default=None,
        help="Optional raw queue column; if missing, queue proxy is derived",
    )
    # Toy arguments
    parser.add_argument(
        "--train-toy",
        action="store_true",
        help="Enable toy IQL training after validation"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200000,
        help="Number of transitions to sample for toy training (after validation)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=20000,
        help="Number of training steps for toy IQL training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=36,
        help="Random seed for sampling and training",
    )
    # Transition build configuration arguments
    parser.add_argument("--history-window", type=int, default=3)
    parser.add_argument("--episode-hours", type=int, default=24)
    parser.add_argument("--action-threshold", type=float, default=0.0)
    parser.add_argument("--deadline-penalty", type=float, default=100.0)
    parser.add_argument("--queue-penalty", type=float, default=0.0)
    parser.add_argument("--execute-penalty", type=float, default=0.0)
    parser.add_argument("--gas-reference-window", type=int, default=128)
    parser.add_argument(
        "--disable-state-normalization",
        action="store_true",
        help="Disable min-max normalization when creating state vectors",
    )
    return parser.parse_args()

def train_toy_iql(dataframe, args) -> int:
    # Placeholder for toy IQL training logic
    pass
    n = min (args.sample_size, len(dataframe))
    if n <= 0:
        print(f"Invalid sample size {args.sample_size}; must be > 0 and less than dataset size.")
        return 1
    
    # 2) sample n transitions from the dataframe then sort by episode_id and step_index to preserve temporal order within episodes
    sampled_df = dataframe.sample(n=n, random_state=args.seed)
    sampled_df = sampled_df.sort_values("timestamp").reset_index(drop=True)

    # 3) Convert the sampled dataframe to a d3rlpy MDPDataset
    dataset = build_d3rlpy_dataset(sampled_df, mode=args.mode)

    # 4) Train a toy IQL agent on the sampled dataset for args.n_steps steps
    try:
        import d3rlpy
    except Exception as exc:
        print(f"d3rlpy is not available. Install d3rlpy in this environment to enable toy training.")
        return 1
    
    iql = d3rlpy.algos.DiscreteCQLConfig().create(device="cpu")

    # 5) fit few steps for santiny check
    iql.fit(dataset, n_steps=args.n_steps)

    out_dir = BASE_DIR.parent / "output" 
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = out_dir / f"toy_iql_{run_id}.d3"
    iql.save_model(model_path)

    print(f"Toy IQL training complete. Model saved to {model_path}")
    return 0


def main() -> int:
    args = parse_args()

    if args.build_from_raw is not None:
        if not args.build_from_raw.exists():
            print(f"Raw parquet not found: {args.build_from_raw}")
            return 1

        config = TransitionBuildConfig(
            action_col=args.action_col,
            queue_col=args.queue_col,
            history_window=args.history_window,
            episode_hours=args.episode_hours,
            action_threshold=args.action_threshold,
            deadline_penalty=args.deadline_penalty,
            queue_penalty=args.queue_penalty,
            execute_penalty=args.execute_penalty,
            gas_reference_window=args.gas_reference_window,
            normalize_state=not args.disable_state_normalization,
        )

        try:
            dataframe = build_transitions_from_parquet(
                args.build_from_raw,
                config,
                output_path=args.output,
            )
        except ValueError as exc:
            print(f"Transition build configuration/data error: {exc}")
            return 1

        print(
            "Built transition dataset from raw parquet "
            f"with {len(dataframe)} rows."
        )
    else:
        if not args.input.exists():
            print(f"Input parquet not found: {args.input}")
            return 1
        dataframe = load_transition_dataframe(args.input)

    result = validate_transition_dataframe(dataframe, mode=args.mode)
    print(format_validation_report(result))

    if not result.passed:
        print("Validation failed. Training is blocked until required transition fields are fixed.")
        return 2

    # Toy training placeholder - this is where you would call your training function after validation passes
    if args.train_toy:
        print("=== Toy Training Config ===")
        print(f"Sample Size: {args.sample_size}")
        print(f"Number of Steps: {args.n_steps}")
        print(f"Seed: {args.seed}")
        print(f"Validation Mode: {args.mode}")
        print("===========================")

        return train_toy_iql(dataframe, args)
    
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
