from __future__ import annotations

import numpy as np
import pandas as pd
import json
from pathlib import Path

from .config import TransitionBuildConfig
from .schema import STATE_COLS, NEXT_STATE_COLS, Q_IDX, T_IDX

REQUIRED_STAGE_COLUMNS = {
    "timestamp", "episode_id", "step_index",
    "action", "gas_t", "queue_size", "time_to_deadline",
    *STATE_COLS,
}


def build_reward_episode_frame(
    state_action_df: pd.DataFrame,
    config: TransitionBuildConfig,
) -> pd.DataFrame:
    """
    Builds the final RL transition frame (s, a, r, s', done) with physically consistent next_states.
    Ensures that next_state reflects all changes (Gas History and Dynamics).
    """
    missing = [column for column in REQUIRED_STAGE_COLUMNS if column not in state_action_df.columns]
    if missing:
        raise ValueError(f"State-action frame is missing required columns: {missing}")

    df = state_action_df.copy()

    # 1. Economic Context
    gas_reference = (
        pd.to_numeric(df["gas_t"], errors="coerce")
        .fillna(0.0)
        .rolling(window=config.gas_reference_window, min_periods=1)
        .mean()
    )

    action_continuous = np.clip(df["action"].to_numpy(dtype=np.float32), 0.0, 1.0)
    pre_execute_queue = df["queue_size"].to_numpy(dtype=np.float32)
    
    # Execution Volume Logic
    execution_capacity = getattr(config, "execution_capacity", 500.0)
    executed_volume_proxy = np.floor(action_continuous * pre_execute_queue)
    executed_volume_proxy = np.clip(executed_volume_proxy, 0, execution_capacity)

    # 2. Economic Reward (Scaled to Gwei)
    gas_scale = getattr(config, "gas_to_gwei_scale", 1e9)
    gas_t_gwei = df["gas_t"].to_numpy(dtype=np.float32) / gas_scale
    gas_ref_gwei = gas_reference.to_numpy(dtype=np.float32) / gas_scale
    
    C_base = config.C_base
    C_mar = config.C_mar

    R_eff = executed_volume_proxy * C_mar * (gas_ref_gwei - gas_t_gwei)
    R_overhead = C_base * gas_t_gwei * (executed_volume_proxy > 0).astype(np.float32)
    
    reward_scale = config.reward_scale
    execution_reward = (R_eff - R_overhead) / reward_scale

    # 3. Urgency Penalty Logic
    beta = config.urgency_beta
    alpha = config.urgency_alpha
    max_time_h = float(config.episode_hours)
    t_deadline_arr = df["time_to_deadline"].to_numpy(dtype=np.float32)
    t_denom = max_time_h * 3600.0 if t_deadline_arr.max() > 24 else max_time_h
    time_ratio = np.clip(t_deadline_arr / max(1e-6, t_denom), 0.0, 1.0)
    
    remaining_q = pre_execute_queue - executed_volume_proxy
    urgency_penalty = (beta / reward_scale) * remaining_q * np.exp(alpha * (1.0 - time_ratio))
    
    total_reward = execution_reward - urgency_penalty

    # 4. Terminal status Check
    is_last_step = (
        df["step_index"]
        == df.groupby("episode_id", dropna=False)["step_index"].transform("max")
    ).to_numpy()
    
    deadline_miss = (is_last_step & (remaining_q > 0))
    total_reward = total_reward - ((config.deadline_penalty / reward_scale) * deadline_miss.astype(np.float32))

    done = is_last_step.astype(np.int8)
    truncated = is_last_step.astype(np.int8)

    # 5. DERIVE PHYSICALLY CONSISTENT NEXT_STATE
    updated_next_states = [None] * len(df)
    
    # Load normalization params ONLY if required by config
    mins, maxs = None, None
    if getattr(config, "normalize_state", False):
        try:
            norm_path = Path("Data/state_norm_params.json")
            if not norm_path.exists():
                 norm_path = Path(__file__).resolve().parent.parent.parent.parent / "Data" / "state_norm_params.json"
            
            if norm_path.exists():
                with open(norm_path, "r") as f:
                    params = json.load(f)
                    mins = np.array(params["mins"], dtype=np.float32)
                    maxs = np.array(params["maxs"], dtype=np.float32)
        except: pass

    Q_IDX, T_IDX = 8, 9
    
    for s_col, ns_col in zip(STATE_COLS, NEXT_STATE_COLS):
        df[ns_col] = df.groupby("episode_id", dropna=False)[s_col].shift(-1)
        
    # Patch Terminal states where shift(-1) is NaN
    terminal_mask = df[NEXT_STATE_COLS[0]].isna()
    for s_col, ns_col in zip(STATE_COLS, NEXT_STATE_COLS):
        df.loc[terminal_mask, ns_col] = df.loc[terminal_mask, s_col]
        
    q_next_terminal = pre_execute_queue[terminal_mask] - executed_volume_proxy[terminal_mask]
    t_next_terminal = 0.0
    
    if mins is not None and maxs is not None:
        denom = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
        q_next_terminal = (q_next_terminal - mins[Q_IDX]) / denom[Q_IDX]
        t_next_terminal = (t_next_terminal - mins[T_IDX]) / denom[T_IDX]
        
    df.loc[terminal_mask, NEXT_STATE_COLS[Q_IDX]] = q_next_terminal
    df.loc[terminal_mask, NEXT_STATE_COLS[T_IDX]] = t_next_terminal

    df_dict = {
        "action": action_continuous,
        "reward": total_reward.astype(np.float32),
        "done": done,
        "episode_id": df["episode_id"].astype(np.int64),
        "timestamp": df["timestamp"],
        "step_index": df["step_index"].astype(np.int64),
        "gas_t": pd.to_numeric(df["gas_t"], errors="coerce").fillna(0.0).astype(np.float32),
        "gas_reference": gas_reference.astype(np.float32),
        "transaction_count": pd.to_numeric(df["transaction_count"], errors="coerce").fillna(0.0).astype(np.float32),
        "queue_size": pd.to_numeric(df["queue_size"], errors="coerce").fillna(0.0).astype(np.float32),
        "time_to_deadline": pd.to_numeric(df["time_to_deadline"], errors="coerce").fillna(0.0).astype(np.float32),
        "executed_volume_proxy": pd.Series(executed_volume_proxy, dtype=np.float32),
        "truncated": truncated,
        "info_json": pd.Series([None] * len(df), dtype="object"),
        "behavior_log_prob": pd.Series([None] * len(df), dtype="object"),
    }
    
    # Add named state and next_state columns
    for s_col, ns_col in zip(STATE_COLS, NEXT_STATE_COLS):
        df_dict[s_col]  = df[s_col]
        df_dict[ns_col] = df[ns_col]
        
    transitions = pd.DataFrame(df_dict)
    return transitions
