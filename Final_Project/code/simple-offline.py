from __future__ import annotations

import json
import pickle
from datetime import datetime
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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
    # Evaluation arguments
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run offline evaluation metrics on a time-based holdout split",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to a trained d3rlpy model (.d3) for RL policy evaluation",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.2,
        help="Holdout ratio for time-based evaluation split (0, 1)",
    )
    parser.add_argument(
        "--baseline-gas-threshold",
        type=float,
        default=None,
        help="Gas threshold for execute-if-low baseline (if gas column exists)",
    )
    parser.add_argument(
        "--cvar-alpha",
        type=float,
        default=0.1,
        help="Alpha for CVaR metric (0, 1)",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help="Optional JSON path to save evaluation report",
    )
    parser.add_argument(
        "--fee-gas-scale",
        type=float,
        default=1e9,
        help="Scale factor for gas unit conversion in fee simulation (default: 1e9, Wei->Gwei)",
    )
    parser.add_argument(
        "--execution-proxy-mode",
        choices=("queue", "unit"),
        default="queue",
        help="Execution volume proxy mode for simulated fee: queue uses queue_size/executed_volume_proxy, unit uses 1 per execute",
    )
    parser.add_argument(
        "--execution-capacity",
        type=float,
        default=None,
        help="Optional cap for execution proxy per step (applied when action=1)",
    )
    parser.add_argument(
        "--no-eval-plots",
        action="store_true",
        help="Disable saving evaluation plots (metrics dashboard and pareto chart)",
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
    except ImportError:
        print("d3rlpy is not available. Install d3rlpy in this environment to enable toy training.")
        return 1
    
    iql = d3rlpy.algos.DiscreteCQLConfig().create(device="cpu")

    # 5) fit few steps for santiny check
    iql.fit(dataset, n_steps=args.n_steps, n_steps_per_epoch=1000)

    out_dir = BASE_DIR.parent / "output" 
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = out_dir / f"toy_iql_{run_id}.d3"
    learnable_path = out_dir / f"toy_iql_{run_id}.d3rl"
    iql.save_model(model_path)
    iql.save(str(learnable_path))

    print(f"Toy IQL training complete. Weights saved to {model_path}")
    print(f"Full learnable checkpoint saved to {learnable_path}")
    return 0


def run_evaluation(dataframe: pd.DataFrame, args: argparse.Namespace) -> int:
    if not (0.0 < args.eval_ratio < 1.0):
        print("eval_ratio must be between 0 and 1 (exclusive).")
        return 1

    if not (0.0 < args.cvar_alpha < 1.0):
        print("cvar_alpha must be between 0 and 1 (exclusive).")
        return 1

    if args.fee_gas_scale <= 0:
        print("fee_gas_scale must be > 0.")
        return 1

    if args.execution_capacity is not None and args.execution_capacity <= 0:
        print("execution_capacity must be > 0 when provided.")
        return 1

    eval_df = _time_holdout_split(dataframe, args.eval_ratio)
    if eval_df.empty:
        print("Evaluation split is empty. Increase dataset size or reduce eval_ratio.")
        return 1

    policies: dict[str, np.ndarray] = {}
    logged_action = pd.to_numeric(eval_df["action"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    policies["logged_policy"] = logged_action
    policies["execute_now"] = np.ones(len(eval_df), dtype=np.int64)

    gas_column = _pick_gas_column(eval_df)
    if gas_column is not None:
        gas_values = pd.to_numeric(eval_df[gas_column], errors="coerce").fillna(0.0)
        threshold = args.baseline_gas_threshold
        if threshold is None:
            threshold = float(gas_values.median())
        threshold_action = (gas_values <= threshold).astype(np.int64).to_numpy()
        policies[f"threshold_policy(gas<={threshold:.6g})"] = threshold_action

    if args.model_path is not None:
        if not args.model_path.exists():
            print(f"Model file not found: {args.model_path}")
            return 1
        try:
            import d3rlpy
        except ImportError as exc:
            print(f"d3rlpy import failed for evaluation: {exc}")
            return 1

        states = np.asarray(eval_df["state"].tolist(), dtype=np.float32)
        model_path_str = str(args.model_path)

        # Try full learnable checkpoint first. If the file was saved via
        # save_model (weights-only), fallback to explicit algo build + load_model.
        algo = None
        try:
            algo = d3rlpy.load_learnable(model_path_str, device="cpu")
        except (pickle.UnpicklingError, ValueError, RuntimeError, OSError, AttributeError, TypeError) as first_exc:
            try:
                bootstrap_n = min(2048, len(eval_df))
                bootstrap_df = eval_df.head(bootstrap_n).copy()
                bootstrap_dataset = build_d3rlpy_dataset(bootstrap_df, mode=args.mode)
                algo = d3rlpy.algos.DiscreteCQLConfig().create(device="cpu")
                algo.build_with_dataset(bootstrap_dataset)
                algo.load_model(model_path_str)
                print(
                    "Loaded weight-only checkpoint via DiscreteCQL fallback "
                    f"(load_learnable failed: {type(first_exc).__name__})."
                )
            except (ValueError, RuntimeError, OSError, AttributeError, TypeError) as second_exc:
                print("Failed to load model for evaluation.")
                print(f"load_learnable error: {first_exc}")
                print(f"weights-only fallback error: {second_exc}")
                return 1

        if algo is None:
            print("Model loading returned no algorithm instance.")
            return 1

        predicted_action = np.asarray(algo.predict(states)).reshape(-1).astype(np.int64)
        policies["rl_policy"] = predicted_action

    results: dict[str, dict[str, float | int | str | None]] = {}
    for policy_name, action_hat in policies.items():
        results[policy_name] = _compute_metrics_for_policy(
            eval_df,
            action_hat,
            cvar_alpha=args.cvar_alpha,
            gas_column=gas_column,
            fee_gas_scale=args.fee_gas_scale,
            execution_proxy_mode=args.execution_proxy_mode,
            execution_capacity=args.execution_capacity,
        )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows_total": int(len(dataframe)),
        "rows_eval": int(len(eval_df)),
        "eval_ratio": float(args.eval_ratio),
        "available_policies": list(results.keys()),
        "metrics": results,
        "skipped_estimators": {
            "WIS": "Skipped: behavior policy probabilities/log-propensity are unavailable.",
            "DR": "Skipped: fitted Q and behavior propensity models are not available in current pipeline.",
            "FQE": "Skipped: dedicated FQE training/evaluation pipeline is not implemented yet.",
        },
    }

    _print_evaluation_report(report)

    output_path = args.report_output
    if output_path is None:
        out_dir = BASE_DIR.parent / "result"
        out_dir.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = out_dir / f"evaluation_{run_id}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Evaluation report saved to {output_path}")

    _save_evaluation_plots(report, output_path, disabled=args.no_eval_plots)

    return 0


def _time_holdout_split(dataframe: pd.DataFrame, eval_ratio: float) -> pd.DataFrame:
    temp = dataframe.copy()
    parsed_ts = pd.to_datetime(temp["timestamp"], utc=True, errors="coerce")

    if parsed_ts.isna().all():
        temp["_order"] = np.arange(len(temp))
        temp = temp.sort_values("_order")
    else:
        temp["_parsed_ts"] = parsed_ts
        temp = temp.sort_values("_parsed_ts", na_position="last")

    split_idx = int((1.0 - eval_ratio) * len(temp))
    eval_df = temp.iloc[split_idx:].copy()
    for col in ["_parsed_ts", "_order"]:
        if col in eval_df.columns:
            eval_df.drop(columns=[col], inplace=True)
    return eval_df.reset_index(drop=True)


def _pick_gas_column(dataframe: pd.DataFrame) -> str | None:
    for col in ["gas_t", "base_fee_per_gas", "gas"]:
        if col in dataframe.columns:
            return col
    return None


def _compute_metrics_for_policy(
    eval_df: pd.DataFrame,
    action_hat: np.ndarray,
    *,
    cvar_alpha: float,
    gas_column: str | None,
    fee_gas_scale: float,
    execution_proxy_mode: str,
    execution_capacity: float | None,
) -> dict[str, float | int | str | None]:
    logged_action = pd.to_numeric(eval_df["action"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    reward = pd.to_numeric(eval_df["reward"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    episode_id = eval_df["episode_id"]

    action_hat = np.asarray(action_hat).reshape(-1).astype(np.int64)
    if len(action_hat) != len(eval_df):
        raise ValueError("Predicted action length does not match evaluation dataframe length.")

    matched = action_hat == logged_action
    coverage = float(matched.mean())

    expected_return_matched = None
    if matched.any():
        expected_return_matched = float(reward[matched].mean())

    action_rate = float((action_hat == 1).mean())

    exec_count_per_episode = pd.Series((action_hat == 1).astype(np.int8)).groupby(episode_id, dropna=False).sum()
    deadline_miss_rate = float((exec_count_per_episode == 0).mean())

    total_cost_per_episode = None
    total_cost_sum = None
    execution_proxy_per_episode = None
    execute_count_per_episode = float(exec_count_per_episode.mean())
    if gas_column is not None:
        gas_value = pd.to_numeric(eval_df[gas_column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        executed_volume_proxy = _derive_execution_proxy(
            eval_df,
            action_hat,
            mode=execution_proxy_mode,
            capacity=execution_capacity,
        )
        gas_converted = gas_value / fee_gas_scale
        step_cost = gas_converted * executed_volume_proxy
        episode_cost = pd.Series(step_cost).groupby(episode_id, dropna=False).sum()
        total_cost_per_episode = float(episode_cost.mean())
        total_cost_sum = float(episode_cost.sum())

        episode_exec_proxy = pd.Series(executed_volume_proxy).groupby(episode_id, dropna=False).sum()
        execution_proxy_per_episode = float(episode_exec_proxy.mean())

    adjusted_return = None
    if expected_return_matched is not None:
        adjusted_return = float(coverage * expected_return_matched)

    matched_episode_return = pd.Series(np.where(matched, reward, 0.0)).groupby(episode_id, dropna=False).sum()
    risk_variance_episode = float(matched_episode_return.var(ddof=0))

    cvar_episode = None
    if len(matched_episode_return) > 0:
        q = float(np.quantile(matched_episode_return.to_numpy(dtype=np.float64), cvar_alpha))
        tail = matched_episode_return[matched_episode_return <= q]
        if len(tail) > 0:
            cvar_episode = float(tail.mean())

    return {
        "coverage": coverage,
        "expected_return_matched": expected_return_matched,
        "action_rate": action_rate,
        "deadline_miss_rate": deadline_miss_rate,
        "total_cost_per_episode": total_cost_per_episode,
        "total_cost_sum": total_cost_sum,
        "execution_proxy_per_episode": execution_proxy_per_episode,
        "fee_gas_scale": float(fee_gas_scale),
        "execution_proxy_mode": execution_proxy_mode,
        "execute_count_per_episode_proxy": execute_count_per_episode,
        "adjusted_return": adjusted_return,
        "reward_variance_episode": risk_variance_episode,
        f"cvar_episode_alpha_{cvar_alpha}": cvar_episode,
    }


def _derive_execution_proxy(
    eval_df: pd.DataFrame,
    action_hat: np.ndarray,
    *,
    mode: str,
    capacity: float | None,
) -> np.ndarray:
    action_mask = (action_hat == 1).astype(np.float64)

    if mode == "unit":
        executed = action_mask
    else:
        if "executed_volume_proxy" in eval_df.columns:
            base_volume = pd.to_numeric(eval_df["executed_volume_proxy"], errors="coerce").fillna(0.0)
        elif "queue_size" in eval_df.columns:
            base_volume = pd.to_numeric(eval_df["queue_size"], errors="coerce").fillna(0.0)
        else:
            base_volume = pd.Series(np.ones(len(eval_df), dtype=np.float64))

        executed = action_mask * base_volume.to_numpy(dtype=np.float64)

    if capacity is not None:
        executed = np.minimum(executed, capacity)

    executed = np.maximum(executed, 0.0)
    return executed


def _print_evaluation_report(report: dict[str, object]) -> None:
    print("=== Evaluation Summary ===")
    print(f"rows_total={report['rows_total']}")
    print(f"rows_eval={report['rows_eval']}")
    print(f"eval_ratio={report['eval_ratio']}")

    metrics = report["metrics"]
    if isinstance(metrics, dict):
        for policy_name, metric in metrics.items():
            print(f"--- {policy_name} ---")
            if isinstance(metric, dict):
                for key, value in metric.items():
                    print(f"{key}: {value}")

    skipped = report.get("skipped_estimators")
    if isinstance(skipped, dict):
        print("--- Skipped Estimators ---")
        for key, reason in skipped.items():
            print(f"{key}: {reason}")


def _save_evaluation_plots(report: dict[str, object], report_path: Path, *, disabled: bool) -> None:
    if disabled:
        print("Evaluation plots are disabled by --no-eval-plots.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available, skip saving evaluation plots.")
        return

    metrics_obj = report.get("metrics")
    if not isinstance(metrics_obj, dict) or len(metrics_obj) == 0:
        print("No policy metrics available, skip plotting.")
        return

    policies = list(metrics_obj.keys())

    def extract_metric(metric_name: str) -> np.ndarray:
        values: list[float] = []
        for policy in policies:
            policy_metrics = metrics_obj.get(policy)
            value = None
            if isinstance(policy_metrics, dict):
                value = policy_metrics.get(metric_name)

            if isinstance(value, (int, float)) and np.isfinite(value):
                values.append(float(value))
            else:
                values.append(np.nan)

        return np.asarray(values, dtype=np.float64)

    metric_specs = [
        ("coverage", "Coverage"),
        ("expected_return_matched", "Expected Return (Matched)"),
        ("adjusted_return", "Adjusted Return"),
        ("action_rate", "Action Rate"),
        ("deadline_miss_rate", "Deadline Miss Rate"),
        ("total_cost_per_episode", "Total Cost Per Episode"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()
    x = np.arange(len(policies))

    for ax, (metric_key, title) in zip(axes_flat, metric_specs):
        values = extract_metric(metric_key)
        ax.bar(x, np.nan_to_num(values, nan=0.0))
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    metrics_plot_path = report_path.with_name(f"{report_path.stem}_metrics.png")
    plt.savefig(metrics_plot_path, dpi=180)
    plt.close(fig)
    print(f"Evaluation metrics plot saved to {metrics_plot_path}")

    cost = extract_metric("total_cost_per_episode")
    cost_label = "Total Cost Per Episode"
    if not np.isfinite(cost).any():
        cost = extract_metric("execute_count_per_episode_proxy")
        cost_label = "Execute Count Per Episode (Proxy)"

    miss = extract_metric("deadline_miss_rate")
    valid = np.isfinite(cost) & np.isfinite(miss)
    if valid.any():
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.scatter(cost[valid], miss[valid], s=60)
        for idx, policy in enumerate(policies):
            if valid[idx]:
                ax2.annotate(policy, (cost[idx], miss[idx]), xytext=(5, 5), textcoords="offset points")

        ax2.set_title("Policy Pareto View")
        ax2.set_xlabel(cost_label)
        ax2.set_ylabel("Deadline Miss Rate")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()

        pareto_plot_path = report_path.with_name(f"{report_path.stem}_pareto.png")
        plt.savefig(pareto_plot_path, dpi=180)
        plt.close(fig2)
        print(f"Evaluation pareto plot saved to {pareto_plot_path}")
    else:
        print("No valid cost/miss values for pareto plot, skipped.")


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

    if args.evaluate:
        return run_evaluation(dataframe, args)

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
