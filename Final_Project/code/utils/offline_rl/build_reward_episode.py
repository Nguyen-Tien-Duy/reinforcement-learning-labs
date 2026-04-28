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

    action_ids = df["action"].to_numpy(dtype=np.int64)
    bins = np.array(config.action_bins, dtype=np.float32)
    action_continuous = bins[action_ids]  # Map bin ID → execution ratio
    pre_execute_queue = df["queue_size"].to_numpy(dtype=np.float32)
    
    # Execution Volume Logic
    execution_capacity = getattr(config, "execution_capacity", 500.0)
    executed_volume_proxy = np.floor(action_continuous * pre_execute_queue)
    executed_volume_proxy = np.clip(executed_volume_proxy, 0, execution_capacity)

    # 2. Efficiency Component (Spec 2.1 & 2.2)
    gas_scale = getattr(config, "gas_to_gwei_scale", 1e9)
    gas_t_gwei = df["gas_t"].to_numpy(dtype=np.float32) / gas_scale
    gas_ref_gwei = gas_reference.to_numpy(dtype=np.float32) / gas_scale
    
    C_base = getattr(config, "C_base", 21000.0)
    s_g = getattr(config, "gas_scaling_factor", 10.0)
    
    # R_eff = [n * (g_ref - g_t) - C_base * g_t * (n > 0)] / s_g
    # This formula creates the "Economies of Scale" incentive.
    has_execution = (executed_volume_proxy > 0).astype(np.float32)
    efficiency_savings = executed_volume_proxy * (gas_ref_gwei - gas_t_gwei)
    overhead_cost = (C_base / 1e9) * gas_t_gwei * has_execution # C_base is in gas units
    
    R_eff = (efficiency_savings - overhead_cost) / s_g

    # 3. Urgency Penalty Component (Spec 2.2)
    beta = getattr(config, "urgency_beta", 0.01)
    alpha = getattr(config, "urgency_alpha", 3.0)
    max_time_h = float(getattr(config, "episode_hours", 24))
    t_deadline_arr = df["time_to_deadline"].to_numpy(dtype=np.float32)
    t_denom = max_time_h * 3600.0 if t_deadline_arr.max() > 24 else max_time_h
    time_ratio = np.clip(t_deadline_arr / max(1e-6, t_denom), 0.0, 1.0)
    
    remaining_q = pre_execute_queue - executed_volume_proxy
    # Linear scaling: maintains high correlation (+0.338).
    # Outliers handled via Robust Normalization (Median/IQR) in trainer.
    R_urg = beta * remaining_q * np.exp(alpha * (1.0 - time_ratio))

    # 4. Catastrophe Component (Spec 2.3)
    is_last_step = (
        df["step_index"]
        == df.groupby("episode_id", dropna=False)["step_index"].transform("max")
    ).to_numpy()
    
    # Triple-Sync V27: Linear penalty — proportional to remaining queue.
    # Gradient preserved: partial cleanup always reduces penalty.
    lambda_d = getattr(config, "deadline_penalty", 100.0)
    R_cat = lambda_d * is_last_step.astype(np.float32) * np.maximum(0.0, remaining_q)
    
    # 5. Final Reward Assembly & Scaling
    reward_scale = getattr(config, "reward_scale", 100.0)
    total_reward = (R_eff - R_urg - R_cat) / reward_scale

    done = is_last_step.astype(np.int8)
    truncated = is_last_step.astype(np.int8)

    # 5. DERIVE PHYSICALLY CONSISTENT NEXT_STATE
    # In V29, we NO LONGER normalize states inside the builder.
    # We preserve raw physical/mathematical units and let the Model Scaler handle it.
    
    Q_IDX, T_IDX = 8, 9
    
    for s_col, ns_col in zip(STATE_COLS, NEXT_STATE_COLS):
        df[ns_col] = df.groupby("episode_id", dropna=False)[s_col].shift(-1)
        
    # Patch Terminal states where shift(-1) is NaN
    terminal_mask = df[NEXT_STATE_COLS[0]].isna()
    for s_col, ns_col in zip(STATE_COLS, NEXT_STATE_COLS):
        df.loc[terminal_mask, ns_col] = df.loc[terminal_mask, s_col]
        
    # For Terminal States, the Next Queue is what's left after execution
    # and Next Time is ALWAYS 0.0 (end of episode).
    q_next_terminal = pre_execute_queue[terminal_mask] - executed_volume_proxy[terminal_mask]
    t_next_terminal = 0.0
    
    df.loc[terminal_mask, NEXT_STATE_COLS[Q_IDX]] = q_next_terminal
    df.loc[terminal_mask, NEXT_STATE_COLS[T_IDX]] = t_next_terminal

    df_dict = {
        "action": action_ids,  # Discrete bin ID (int64)
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
    
    # V21: Preserve policy labeling for audit and oracle-only training
    if "policy_type" in df.columns:
        df_dict["policy_type"] = df["policy_type"]
    
    # Add named state and next_state columns
    for s_col, ns_col in zip(STATE_COLS, NEXT_STATE_COLS):
        df_dict[s_col]  = df[s_col]
        df_dict[ns_col] = df[ns_col]
        
    transitions = pd.DataFrame(df_dict)
    return transitions
