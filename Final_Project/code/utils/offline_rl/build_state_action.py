from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TransitionBuildConfig


def build_state_action_frame(raw_df: pd.DataFrame, config: TransitionBuildConfig) -> pd.DataFrame:
    # Validate the inputs is contain timestampt and gas columns
    _validate_state_action_inputs(raw_df, config)

    df = raw_df.copy()
    df[config.timestamp_col] = pd.to_datetime(df[config.timestamp_col], utc=True, errors="coerce")
    if df[config.timestamp_col].isna().any():
        raise ValueError("timestamp column contains invalid values that cannot be parsed.")
    # Sort the dataframe by timestamp and reset the index
    df = df.sort_values(config.timestamp_col).reset_index(drop=True)

    # Get the episode id for each row
    min_ts = df[config.timestamp_col].min() # Get the minimum timestamp in the dataframe
    horizon_seconds = int(config.episode_hours * 3600) # Set the horizon in seconds
    elapsed_seconds = (df[config.timestamp_col] - min_ts).dt.total_seconds().astype(np.int64) # Get the elapsed seconds from the minimum timestamp
    episode_id = (elapsed_seconds // horizon_seconds).astype(np.int64) # Get the episode id for each row

    # Derive the action, queue size, and time to deadline
    action = _derive_action(df, config)
    queue_size = _derive_queue(df, episode_id, action, config)
    time_to_deadline = _derive_time_to_deadline(df, episode_id, config)

    # Build the state matrix
    # Get the first gas price for each episode
    first_gas_price = df[config.gas_col].groupby(episode_id, dropna=False).transform("first")
    gas_history_cols: list[pd.Series] = []
    grouped_gas = df.groupby(episode_id, dropna=False)[config.gas_col]
    for lag in range(config.history_window):
        lag_col = grouped_gas.shift(lag)
        # We prevent data leak when fill some fist null
        lag_col = lag_col.fillna(first_gas_price)
        # Fill the rest of the null values with the previous value
        lag_col = lag_col.groupby(episode_id, dropna=False).ffill()
        gas_history_cols.append(pd.to_numeric(lag_col, errors="coerce").fillna(0.0))

    # ---- PSEUDO-MEMPOOL FEATURES (from MEMPOOL_APPROX_AND_READING.md) ----

    # 1) p_t: Block congestion pressure (EIP-1559 mechanism)
    #    p_t = (gas_used - target_gas) / target_gas, where target_gas = gas_limit / 2
    #    p_t > 0 means congestion, p_t < 0 means low pressure
    gas_used = pd.to_numeric(df[config.gas_used_col], errors="coerce").fillna(0.0)
    gas_limit = pd.to_numeric(df[config.gas_limit_col], errors="coerce").fillna(1.0)
    target_gas = gas_limit / 2.0
    p_t = ((gas_used - target_gas) / target_gas.replace(0, 1)).fillna(0.0)

    # 2) m_t: Base fee momentum (log return between consecutive blocks)
    #    m_t = log(baseFee_t) - log(baseFee_{t-1})
    #    Positive momentum = rising demand pressure
    gas_price = pd.to_numeric(df[config.gas_col], errors="coerce").fillna(1.0).clip(lower=1.0)
    log_gas = np.log(gas_price)
    m_t = log_gas.groupby(episode_id, dropna=False).diff().fillna(0.0)

    # 3) a_t: Base fee acceleration (second derivative)
    #    a_t = m_t - m_{t-1}
    a_t = m_t.groupby(episode_id, dropna=False).diff().fillna(0.0)

    # 4) u_t: Transaction surprise (zscore of deviation from moving average)
    #    Captures bursts beyond local baseline
    tx_count = pd.to_numeric(df[config.transaction_count_col], errors="coerce").fillna(0.0)
    tx_ma = tx_count.groupby(episode_id, dropna=False).transform(
        lambda x: x.rolling(window=128, min_periods=1).mean()
    )
    tx_deviation = tx_count - tx_ma
    tx_std = tx_count.groupby(episode_id, dropna=False).transform(
        lambda x: x.rolling(window=128, min_periods=1).std().fillna(1.0).replace(0, 1)
    )
    u_t = (tx_deviation / tx_std).fillna(0.0)

    # 5) b_t: Latent backlog proxy (pseudo-mempool hidden state estimate)
    #    b_t = max(0, rho * b_{t-1} + alpha * p_t + beta * u_t)
    #    Coefficients are fixed during experiment
    rho, alpha, beta = 0.95, 0.3, 0.2
    b_values = np.zeros(len(df), dtype=np.float64)
    prev_ep = None
    for i in range(len(df)):
        ep = episode_id.iloc[i]
        if ep != prev_ep:
            b_values[i] = 0.0
            prev_ep = ep
        else:
            b_values[i] = max(0.0, rho * b_values[i - 1] + alpha * p_t.iloc[i] + beta * u_t.iloc[i])
    b_t = pd.Series(b_values, index=df.index)

    # Build state matrix: [gas_t, gas_t-1, gas_t-2, p_t, m_t, a_t, u_t, b_t, queue, time]
    state_matrix = np.column_stack(
        [
            *(series.to_numpy(dtype=np.float32) for series in gas_history_cols),
            p_t.to_numpy(dtype=np.float32),
            m_t.to_numpy(dtype=np.float32),
            a_t.to_numpy(dtype=np.float32),
            u_t.to_numpy(dtype=np.float32),
            b_t.to_numpy(dtype=np.float32),
            queue_size.to_numpy(dtype=np.float32),
            time_to_deadline.to_numpy(dtype=np.float32),
        ]
    )

    if config.normalize_state:
        mins = state_matrix.min(axis=0)
        maxs = state_matrix.max(axis=0)
        # Prevent devide by 0 when all values are the same
        denom = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
        state_matrix = (state_matrix - mins) / denom

    state_series = pd.Series(
        [row.astype(np.float32).tolist() for row in state_matrix],
        index=df.index,
        name="state",
    )

    step_index = df.groupby(episode_id, dropna=False).cumcount().astype(np.int64)

    return pd.DataFrame(
        {
            "timestamp": df[config.timestamp_col],
            "episode_id": episode_id,
            "step_index": step_index,
            "state": state_series,
            "action": action.astype(np.float32),
            "gas_t": pd.to_numeric(df[config.gas_col], errors="coerce").fillna(0.0),
            "queue_size": queue_size.astype(np.float32),
            "time_to_deadline": time_to_deadline.astype(np.float32),
        }
    )


def _validate_state_action_inputs(raw_df: pd.DataFrame, config: TransitionBuildConfig) -> None:
    """This function validates the inputs to the build_state_action function.
    
    Args:
        raw_df: The raw dataframe containing the data.
        config: The configuration for the state building.

    How this work: it's check if the required columns (timestamp and gas) are present in the raw dataframe.
    """
    
    required_columns = {config.timestamp_col, config.gas_col}
    missing = [column for column in required_columns if column not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing required raw columns for state building: {missing}")

    if config.history_window < 1:
        raise ValueError("history_window must be >= 1")

    if config.episode_hours < 1:
        raise ValueError("episode_hours must be >= 1")



def _derive_action(df: pd.DataFrame, config: TransitionBuildConfig) -> pd.Series:
    """This function derives the action from the raw dataframe.
    
    Args:
        df: The raw dataframe containing the data.
        config: The configuration for the state building.

    How this work: it's check if the action_col is present in the raw dataframe. If it
    is present, it will derive the action from the action_col. If it is not present,
    it will raise a ValueError.
    """
    if config.action_col is None:
        raise ValueError(
            "action_col is not configured. Provide historical action/sent column to build offline RL transitions."
        )

    if config.action_col not in df.columns:
        raise ValueError(
            f"Configured action_col '{config.action_col}' does not exist in raw dataframe."
        )

    action_source = pd.to_numeric(df[config.action_col], errors="coerce")
    if action_source.isna().all():
        raise ValueError(
            f"action_col '{config.action_col}' has no numeric values; cannot derive discrete action."
        )
    # We cast to continuous float type for Continuous RL formulations
    return (action_source.fillna(0.0) > config.action_threshold).astype(np.float32)


def _derive_queue(
    df: pd.DataFrame,
    episode_id: pd.Series,
    action: pd.Series,
    config: TransitionBuildConfig,
) -> pd.Series:
    """This function derives the queue from the raw dataframe.
    
    Args:
        df: The raw dataframe containing the data.
        episode_id: The episode id for each row.
        action: The action for each row.
        config: The configuration for the state building.

    How this work: it's check if the queue_col is present in the raw dataframe. If it
    is present, it will derive the queue from the queue_col. If it is not present,
    it will derive the queue from the action column.
    """
    if config.queue_col and config.queue_col in df.columns:
        queue = pd.to_numeric(df[config.queue_col], errors="coerce").fillna(0.0)
        return queue.clip(lower=0.0)

    wait_flag = (action == 0).astype(np.int8)
    execute_count = action.groupby(episode_id, dropna=False).cumsum()
    queue_proxy = wait_flag.groupby([episode_id, execute_count], dropna=False).cumsum()
    return queue_proxy.astype(np.float32)



def _derive_time_to_deadline(
    df: pd.DataFrame,
    episode_id: pd.Series,
    config: TransitionBuildConfig,
) -> pd.Series:
    """This function derives the time to deadline from the raw dataframe.
    
    Args:
        df: The raw dataframe containing the data.
        episode_id: The episode id for each row.
        config: The configuration for the state building.

    How this work: it's calculate the time difference between the current timestamp and the maximum timestamp in the episode.
    """
    episode_end = df.groupby(episode_id, dropna=False)[config.timestamp_col].transform("max")
    remaining_hours = (episode_end - df[config.timestamp_col]).dt.total_seconds() / 3600.0
    return remaining_hours.clip(lower=0.0)
