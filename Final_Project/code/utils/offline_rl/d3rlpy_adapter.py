from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from .types import TransitionValidationError, ValidationMode
from .validation import validate_transition_dataframe


def build_d3rlpy_dataset(df: pd.DataFrame, *, mode: ValidationMode = "strict") -> Any:
    """Build a d3rlpy MDPDataset after strict validation."""
    result = validate_transition_dataframe(df, mode=mode)
    if result.errors:
        raise TransitionValidationError(result)

    try:
        from d3rlpy.dataset import MDPDataset
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "d3rlpy is not available. Install d3rlpy in this environment to build dataset objects."
        ) from exc

    observations = _to_state_array(df["state"])
    rewards = pd.to_numeric(df["reward"], errors="raise").to_numpy(dtype=np.float32)
    terminals = _to_terminal_array(df["done"])
    actions = _to_action_array(df["action"])

    try:
        return MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
        )
    except TypeError:
        # Compatibility fallback for older/newer d3rlpy signatures.
        return MDPDataset(observations, actions, rewards, terminals)


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


def _to_action_array(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.to_numpy(dtype=np.float32)

    arrays: list[np.ndarray] = []
    expected_shape: tuple[int, ...] | None = None
    for idx, value in series.items():
        arr = _parse_array_like(value)
        if expected_shape is None:
            expected_shape = arr.shape
        if arr.shape != expected_shape:
            raise ValueError(
                f"Action shape mismatch at row {idx}: got {arr.shape}, expected {expected_shape}."
            )
        arrays.append(arr)
    return np.stack(arrays, axis=0)


def _to_terminal_array(series: pd.Series) -> np.ndarray:
    if is_bool_dtype(series):
        return series.to_numpy(dtype=bool)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError("Column 'done' contains values that cannot be converted to bool.")
    return numeric.astype(np.int64).isin([1]).to_numpy(dtype=bool)
