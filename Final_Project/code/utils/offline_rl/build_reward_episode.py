from __future__ import annotations

import numpy as np
import pandas as pd
import json
from pathlib import Path

from .config import TransitionBuildConfig

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
    
    # Iterate by Episode to correctly handle shifts and boundaries
    for episode_id, group in df.groupby("episode_id", sort=False):
        indices = group.index.tolist()
        for i_local, idx in enumerate(indices):
            # TRANSITION RECONSTRUCTION
            q_t = float(group.loc[idx, "queue_size"])
            a_t = action_continuous[idx]
            exec_v = min(np.floor(a_t * q_t), execution_capacity)
            
            if i_local < len(indices) - 1:
                next_row_idx = indices[i_local + 1]
                # PULL NEXT observation vector directly to get converged gas history
                s_next = np.array(group.loc[next_row_idx, "state"], dtype=np.float32).copy()
                
                # Patch in dynamic physics
                arrival_next = float(group.loc[next_row_idx, "transaction_count"])
                t_next = float(group.loc[next_row_idx, "time_to_deadline"])
                q_next = q_t - exec_v + arrival_next
            else:
                # Terminal step: stays on current state but zeroes time and finalizes queue
                s_next = np.array(group.loc[idx, "state"], dtype=np.float32).copy()
                q_next = q_t - exec_v
                t_next = 0.0
            
            # Application of injection values (normalized or raw)
            if mins is not None and maxs is not None:
                denom = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
                s_next[Q_IDX] = (q_next - mins[Q_IDX]) / denom[Q_IDX]
                s_next[T_IDX] = (t_next - mins[T_IDX]) / denom[T_IDX]
            else:
                s_next[Q_IDX] = q_next
                s_next[T_IDX] = t_next
            
            updated_next_states[idx] = s_next.tolist()

    next_state = pd.Series(updated_next_states, index=df.index)

    transitions = pd.DataFrame(
        {
            "state": df["state"],
            "action": action_continuous,
            "reward": total_reward.astype(np.float32),
            "next_state": next_state,
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
    )
    return transitions
