import numpy as np
import pandas as pd
import json
from pathlib import Path

from .config import TransitionBuildConfig
from .schema import STATE_COLS, Q_IDX


def build_state_action_frame(raw_df: pd.DataFrame, config: TransitionBuildConfig) -> pd.DataFrame:
    _validate_state_action_inputs(raw_df, config)

    df = raw_df.copy()
    df[config.timestamp_col] = pd.to_datetime(df[config.timestamp_col], utc=True, errors="coerce")
    if df[config.timestamp_col].isna().any():
        raise ValueError("timestamp column contains invalid values that cannot be parsed.")
    
    df = df.sort_values(config.timestamp_col).reset_index(drop=True)

    min_ts = df[config.timestamp_col].min()
    horizon_seconds = int(config.episode_hours * 3600)
    elapsed_seconds = (df[config.timestamp_col] - min_ts).dt.total_seconds().astype(np.int64)
    episode_id = (elapsed_seconds // horizon_seconds).astype(np.int64)

    # 1. Action: Continuous ratio [0, 1] of block gas limit
    action = _derive_action(df, config)
    
    # 2. Queue: Physically consistent model
    # Q[t] = CumArrivals[t] - CumExecuted[t-1]
    queue_size = _derive_queue(df, episode_id, action, config)
    
    # 3. Time: Remaining hours in episode
    time_to_deadline = _derive_time_to_deadline(df, episode_id, config)

    # 4. Build Observation Vector (11 dimensions)
    # [3xHistory, Congestion, Momentum, Acceleration, Surprise, Backlog, Queue, Time, GasRef]
    
    first_gas_price = df[config.gas_col].groupby(episode_id, dropna=False).transform("first")
    gas_history_cols = []
    grouped_gas = df.groupby(episode_id, dropna=False)[config.gas_col]
    for lag in range(config.history_window):
        lag_col = grouped_gas.shift(lag).fillna(first_gas_price).groupby(episode_id, dropna=False).ffill()
        gas_history_cols.append(pd.to_numeric(lag_col, errors="coerce").fillna(0.0))

    # Features: No more silent fillers. Data MUST be clean.
    gas_used = pd.to_numeric(df[config.gas_used_col], errors="coerce")
    _ensure_no_nan(gas_used, config.gas_used_col)
    
    gas_limit = pd.to_numeric(df[config.gas_limit_col], errors="coerce").replace(0, np.nan)
    _ensure_no_nan(gas_limit, config.gas_limit_col)
    target_gas = gas_limit / 2.0
    p_t = ((gas_used - target_gas) / target_gas).fillna(0.0)

    gas_price = pd.to_numeric(df[config.gas_col], errors="coerce").clip(lower=1.0)
    _ensure_no_nan(gas_price, config.gas_col)
    
    log_gas = np.log(gas_price)
    m_t = log_gas.groupby(episode_id, dropna=False).diff().fillna(0.0) # diff fillna is math-required for first step
    a_t = m_t.groupby(episode_id, dropna=False).diff().fillna(0.0)

    tx_count = pd.to_numeric(df[config.transaction_count_col], errors="coerce")
    _ensure_no_nan(tx_count, config.transaction_count_col)
    
    tx_ma = tx_count.groupby(episode_id, dropna=False).transform(lambda x: x.rolling(window=128, min_periods=1).mean())
    tx_std = tx_count.groupby(episode_id, dropna=False).transform(lambda x: x.rolling(window=128, min_periods=1).std().fillna(1.0).replace(0, 1))
    u_t = ((tx_count - tx_ma) / tx_std)
    _ensure_no_nan(u_t, "calculated u_t (Surprise)")

    rho, alpha_b, beta_b = 0.95, 0.3, 0.2
    b_values = np.zeros(len(df), dtype=np.float64)
    prev_ep = None
    for i in range(len(df)):
        ep = episode_id.iloc[i]
        if ep != prev_ep:
            b_values[i] = 0.0
            prev_ep = ep
        else:
            b_values[i] = max(0.0, rho * b_values[i-1] + alpha_b * p_t.iloc[i] + beta_b * u_t.iloc[i])
    b_t = pd.Series(b_values, index=df.index)

    gas_ref = gas_price.groupby(episode_id, dropna=False).transform(lambda x: x.rolling(window=128, min_periods=1).mean())

    state_matrix = np.stack([
        *(series.to_numpy(dtype=np.float32) for series in gas_history_cols),
        p_t.to_numpy(dtype=np.float32),
        m_t.to_numpy(dtype=np.float32),
        a_t.to_numpy(dtype=np.float32),
        u_t.to_numpy(dtype=np.float32),
        b_t.to_numpy(dtype=np.float32),
        queue_size.to_numpy(dtype=np.float32),
        time_to_deadline.to_numpy(dtype=np.float32),
        gas_ref.to_numpy(dtype=np.float32),
    ], axis=1)


    df_dict = {
        "timestamp": df[config.timestamp_col],
        "episode_id": episode_id,
        "step_index": df.groupby(episode_id).cumcount(),
        "action": action.astype(np.int32),
        "gas_t": gas_price.astype(np.float32),
        "transaction_count": tx_count.astype(np.float32),
        "queue_size": queue_size.astype(np.float32),
        "time_to_deadline": time_to_deadline.astype(np.float32),
    }

    # Flatten state into named float32 columns (STATE_COLS is the single source of truth)
    for i, col in enumerate(STATE_COLS):
        df_dict[col] = pd.Series(state_matrix[:, i].astype(np.float32), index=df.index)

    return pd.DataFrame(df_dict)


def _derive_action(df, config):
    """Quantize raw gas_used/gas_limit ratio into discrete bin IDs."""
    if config.action_col not in df.columns:
        # If building from raw, we might not have 'action' yet. Default to 0.
        return pd.Series(0, index=df.index, dtype=np.int64)
        
    action_source = pd.to_numeric(df[config.action_col], errors="coerce").fillna(0.0)
    gas_limit = pd.to_numeric(df.get(config.gas_limit_col, 30e6), errors="coerce").fillna(30e6).replace(0, 30e6)
    ratio = (action_source / gas_limit).clip(0, 1).to_numpy(dtype=np.float32)
    bins = np.array(config.action_bins, dtype=np.float32)
    # Find nearest bin for each ratio value
    bin_ids = np.argmin(np.abs(ratio[:, None] - bins[None, :]), axis=1)
    return pd.Series(bin_ids, index=df.index, dtype=np.int64)


def _derive_queue(df, episode_id, action, config):

    arrivals = pd.to_numeric(df[config.transaction_count_col], errors="coerce").fillna(0.0).to_numpy()
    arrival_scale = float(getattr(config, "arrival_scale", 0.5))
    # DISCRETE PHYSICS: Round arrivals to nearest integer
    arrivals = np.round(arrivals * arrival_scale).astype(np.int64)
    
    action_ids = action.to_numpy(dtype=np.int64)
    eps = episode_id.to_numpy()
    bins = np.array(config.action_bins, dtype=np.float32)
    
    queues = np.zeros(len(df), dtype=np.int64)
    exec_cap = int(float(getattr(config, "execution_capacity", 500.0)))
    
    current_q = 0
    prev_ep = None
    
    for i in range(len(df)):
        if eps[i] != prev_ep:
            current_q = arrivals[i]
            prev_ep = eps[i]
            
        queues[i] = current_q
        
        # Map bin ID to execution ratio
        ratio = bins[action_ids[i]]
        executed = int(min(np.floor(ratio * current_q), exec_cap))
        
        if i < len(df) - 1 and eps[i+1] == eps[i]:
            arrival_next = arrivals[i+1]
            current_q = max(0, current_q - executed + arrival_next)
            
    return pd.Series(queues, index=df.index, dtype=np.int64)


def _ensure_no_nan(series, col_name):
    if series.isna().any():
        n_missing = series.isna().sum()
        raise ValueError(f"Strict Data Policy Violation: Column '{col_name}' contains {n_missing} NaN values. Please clean your data before building transitions.")


def _derive_time_to_deadline(df, episode_id, config):
    ep_end = df.groupby(episode_id)[config.timestamp_col].transform("max")
    return (ep_end - df[config.timestamp_col]).dt.total_seconds() / 3600.0


def recalculate_queue_and_state(df: pd.DataFrame, config: TransitionBuildConfig) -> pd.DataFrame:
    """
    Recalculate queue_size and patch state vector after Oracle has overwritten the action column.
    Ensures Normalization Integrity by using the ORIGINAL physical limits for the new state.
    """
    df = df.copy()
    episode_id = df["episode_id"]
    
    # 1. Recalculate physical queue with the NEW (Oracle-mixed) discrete actions
    new_queue = _derive_queue(df, episode_id, df["action"], config)
    df["queue_size"] = new_queue.astype(np.float32)
    
    # 2. Patch state vector at Q_IDX=8
    Q_IDX = 8
    queue_values = new_queue.to_numpy(dtype=np.float32)
    
    df[STATE_COLS[Q_IDX]] = queue_values
    
    print(f"[+] Recalculated queue dynamics for {episode_id.nunique()} episodes.")
    print(f"    New Queue range: [{queue_values.min():.1f}, {queue_values.max():.1f}]")
    
    return df


def _validate_state_action_inputs(raw_df, config):
    req = {config.timestamp_col, config.gas_col}
    missing = [c for c in req if c not in raw_df.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")

