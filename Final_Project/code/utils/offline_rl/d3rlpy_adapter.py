from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from .types import TransitionValidationError, ValidationMode
from .validation import validate_transition_dataframe
from .schema import STATE_COLS, NEXT_STATE_COLS


def build_d3rlpy_dataset(df: pd.DataFrame, *, mode: ValidationMode = "strict") -> Any:
    """
    Build a d3rlpy MDPDataset with full Terminals vs Timeouts differentiation.
    Ensures that IQL can accurately bootstrap at episode boundaries.
    """
    result = validate_transition_dataframe(df, mode=mode)
    if result.errors:
        raise TransitionValidationError(result)

    from d3rlpy import ActionSpace
    from d3rlpy.dataset import MDPDataset

    observations = df[STATE_COLS].to_numpy(dtype=np.float32)
    rewards = pd.to_numeric(df["reward"], errors="raise").to_numpy(dtype=np.float32)
    
    # 1. Terminals: Episode ends because of a goal or failure (SLA miss)
    terminals = _to_bool_array(df["done"])
    
    # 2. Timeouts (Truncated): Episode ends because maximum steps reached
    if "truncated" in df.columns:
        timeouts = _to_bool_array(df["truncated"])
    else:
        timeouts = np.zeros(len(df), dtype=bool)
    
    # CRITICAL: d3rlpy v2.x does not allow both to be True at the same time.
    # We prioritize timeouts for episode boundaries to allow bootstrapping.
    terminals = terminals & ~timeouts

    # 3. Actions: Discrete bin IDs (V6)
    actions = df["action"].to_numpy(dtype=np.int64)

    # [SOTA OPTIMIZATION] Memory Alignment & Contiguity
    # Giúp giảm L1/LLC Cache Misses bằng cách ép dữ liệu nằm liên tiếp trong RAM
    observations = np.ascontiguousarray(observations, dtype=np.float32)
    actions = np.ascontiguousarray(actions.reshape(-1, 1), dtype=np.int64)
    rewards = np.ascontiguousarray(rewards.reshape(-1, 1), dtype=np.float32)
    terminals = np.ascontiguousarray(terminals, dtype=np.uint8)
    timeouts = np.ascontiguousarray(timeouts, dtype=np.uint8)

    print(f"[⚡] Memory Optimized: Arrays are now C-contiguous. Ready to saturate CPU cache.")

    return MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
        action_space=ActionSpace.DISCRETE
    )

def _parse_array_like(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=np.float32)
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith("[") or text.startswith("("):
            arr = np.asarray(ast.literal_eval(text), dtype=np.float32)
        else:
            arr = np.asarray([float(text)], dtype=np.float32)
    else:
        arr = np.asarray([value], dtype=np.float32)

    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.astype(np.float32)

def _to_state_array(series: pd.Series) -> np.ndarray:
    arrays: list[np.ndarray] = []
    expected_shape: tuple[int, ...] | None = None

    for idx, value in series.items():
        arr = _parse_array_like(value)
        if expected_shape is None:
            expected_shape = arr.shape
        if arr.shape != expected_shape:
            raise ValueError(
                f"State shape mismatch at row {idx}: got {arr.shape}, expected {expected_shape}."
            )
        arrays.append(arr)

    return np.stack(arrays, axis=0)


def _to_bool_array(series: pd.Series) -> np.ndarray:
    """Helper to convert various types to boolean numpy array. Strictly refuses NaNs."""
    if is_bool_dtype(series):
        return series.to_numpy(dtype=bool)

    # Force raise on any conversion error
    numeric = pd.to_numeric(series, errors="raise")
    
    if numeric.isna().any():
        raise ValueError(f"Strict Data Policy Violation: Boolean column contains NaNs. Policy requires 100% clean data.")
    
    return numeric.astype(np.int64).isin([1]).to_numpy(dtype=bool)
