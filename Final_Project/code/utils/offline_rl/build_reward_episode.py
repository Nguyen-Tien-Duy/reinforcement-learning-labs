from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TransitionBuildConfig

# We define the required columns for the state-action frame
REQUIRED_STAGE_COLUMNS = {
    "timestamp",
    "episode_id",
    "step_index",
    "state",
    "action",
    "gas_t",
    "queue_size",
    "time_to_deadline",
}


def build_reward_episode_frame(
    state_action_df: pd.DataFrame,
    config: TransitionBuildConfig,
) -> pd.DataFrame:
    # Check if the state-action frame is missing required columns
    missing = [column for column in REQUIRED_STAGE_COLUMNS if column not in state_action_df.columns]
    if missing:
        raise ValueError(f"State-action frame is missing required columns: {missing}")

    df = state_action_df.copy()

    # Calculate the gas reference
    gas_reference = (
        pd.to_numeric(df["gas_t"], errors="coerce")
        .fillna(0.0)
        .rolling(window=config.gas_reference_window, min_periods=1)
        .mean()
    )

    # Action is now a continuous float [0.0, 1.0]
    action_continuous = np.clip(df["action"].to_numpy(dtype=np.float32), 0.0, 1.0)
    
    # Get the deadline hit flag
    # deadline_hit = (df["time_to_deadline"] <= 0.0).to_numpy(dtype=np.int8)

    # Shift back queue_size 1 step
    pre_execute_queue = (
        df.groupby("episode_id", dropna=False)["queue_size"]
        .shift(1)
        .fillna(1.0)  # First step has no prior history, default to 1
    ).to_numpy(dtype=np.float32)

    # Calculate actual execution volume (step-1 * queue_size)
    executed_volume_proxy = np.floor(action_continuous * pre_execute_queue)

    # Continuous Economy of Scale Reward natively scaled to Gwei
    gas_t_gwei = df["gas_t"].to_numpy(dtype=np.float32) / getattr(config, "gas_to_gwei_scale", 1e9)
    gas_ref_gwei = gas_reference.to_numpy(dtype=np.float32) / getattr(config, "gas_to_gwei_scale", 1e9)
    
    C_base = getattr(config, "C_base", 21000.0)
    C_mar = getattr(config, "C_mar", 15000.0)

    # Efficiency reward based on C_mar gas equivalence (marginal volume savings natively scaled)
    R_eff = executed_volume_proxy * C_mar * (gas_ref_gwei - gas_t_gwei)
    # Overhead strictly penalizing if making transaction
    R_overhead = C_base * gas_t_gwei * (executed_volume_proxy > 0).astype(np.float32)

    execution_reward = R_eff - R_overhead

    # Component 2: Urgency (cost of delay, risk-averse near deadline)
    beta = getattr(config, "urgency_beta", 0.01)
    alpha = getattr(config, "urgency_alpha", 3.0)
    max_time = float(config.episode_hours)
    time_ratio = np.clip(df["time_to_deadline"].to_numpy(dtype=np.float32) / max_time, 0.0, 1.0)
    
    urgency_penalty = beta * df["queue_size"].to_numpy(dtype=np.float32) * np.exp(alpha * (1.0 - time_ratio))

    reward = execution_reward - urgency_penalty

    # Component 3: Catastrophe (hard SLA violation)
    is_last_step = (
        df["step_index"]
        == df.groupby("episode_id", dropna=False)["step_index"].transform("max")
    )
    # deadline miss occurs when we reach the end and still have people waiting 
    deadline_miss = (is_last_step & (df["queue_size"].to_numpy(dtype=np.float32) > 0)).to_numpy(dtype=np.int8)
    reward = reward - (config.deadline_penalty * deadline_miss)

    # REWARD SCALING: Dividing by 1e6 to bring the range from Millions to roughly [-15, 15] 
    # for numerical stability in the neural network.
    reward_scale = getattr(config, "reward_scale", 1e6)
    reward = reward / reward_scale

    # Done only signals episode end boundary under continuous formulation
    done = is_last_step.to_numpy(dtype=np.int8)
    truncated = is_last_step.to_numpy(dtype=np.int8)

    next_state = df.groupby("episode_id", dropna=False)["state"].shift(-1)
    next_state = next_state.where(next_state.notna(), df["state"])

    transitions = pd.DataFrame(
        {
            "state": df["state"],
            "action": action_continuous,
            "reward": reward.astype(np.float32),
            "next_state": next_state,
            "done": done.astype(np.int8),
            "episode_id": df["episode_id"].astype(np.int64),
            "timestamp": df["timestamp"],
            "step_index": df["step_index"].astype(np.int64),
            "gas_t": pd.to_numeric(df["gas_t"], errors="coerce").fillna(0.0).astype(np.float32),
            "queue_size": pd.to_numeric(df["queue_size"], errors="coerce").fillna(0.0).astype(np.float32),
            "time_to_deadline": pd.to_numeric(df["time_to_deadline"], errors="coerce").fillna(0.0).astype(np.float32),
            "executed_volume_proxy": pd.Series(executed_volume_proxy, dtype=np.float32),
            "truncated": truncated.astype(np.int8),
            "info_json": pd.Series([None] * len(df), dtype="object"),
            "behavior_log_prob": pd.Series([None] * len(df), dtype="object"),
        }
    )
    return transitions
