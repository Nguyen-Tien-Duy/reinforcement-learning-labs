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

    # Get the execute flag
    execute_flag = (df["action"].astype(np.int8) == 1).to_numpy(dtype=np.int8)
    # Get the deadline hit flag
    deadline_hit = (df["time_to_deadline"] <= 0.0).to_numpy(dtype=np.int8)

    # Calculate the execution reward
    execution_reward = execute_flag * (gas_reference.to_numpy(dtype=np.float32) - df["gas_t"].to_numpy(dtype=np.float32))
    # Calculate the reward
    reward = (
        execution_reward
        - (config.execute_penalty * execute_flag)
        - (config.queue_penalty * df["queue_size"].to_numpy(dtype=np.float32))
    )

    is_last_step = (
        df["step_index"]
        == df.groupby("episode_id", dropna=False)["step_index"].transform("max")
    )
    deadline_miss = (is_last_step & (execute_flag == 0) & (deadline_hit == 1)).to_numpy(dtype=np.int8)
    reward = reward - (config.deadline_penalty * deadline_miss)

    done = ((execute_flag == 1) | is_last_step.to_numpy(dtype=bool)).astype(np.int8)
    truncated = (is_last_step.to_numpy(dtype=bool) & (execute_flag == 0)).astype(np.int8)

    next_state = df.groupby("episode_id", dropna=False)["state"].shift(-1)
    next_state = next_state.where(next_state.notna(), df["state"])

    executed_volume_proxy = np.where(
        df["action"].astype(np.int8).to_numpy() == 1,
        df["queue_size"].to_numpy(dtype=np.float32),
        0.0,
    )

    transitions = pd.DataFrame(
        {
            "state": df["state"],
            "action": df["action"].astype(np.int8),
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
