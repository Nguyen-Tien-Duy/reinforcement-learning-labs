from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

ValidationMode = Literal["strict", "lenient"]


@dataclass(frozen=True)
class ColumnRule:
    allowed_dtype_categories: tuple[str, ...]
    nullable: bool = False


@dataclass(frozen=True)
class TransitionSchema:
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...]
    column_rules: dict[str, ColumnRule]


@dataclass
class ValidationIssue:
    code: str
    severity: Literal["error", "warning"]
    message: str
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    sample_rows: list[int] = field(default_factory=list)
    remediation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "columns": self.columns,
            "row_count": self.row_count,
            "sample_rows": self.sample_rows,
            "remediation": self.remediation,
        }


@dataclass
class ValidationResult:
    passed: bool
    mode: ValidationMode
    schema_name: str
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "mode": self.mode,
            "schema_name": self.schema_name,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "summary": self.summary,
        }


class TransitionValidationError(RuntimeError):
    def __init__(self, result: ValidationResult):
        self.result = result
        error_codes = ", ".join(issue.code for issue in result.errors) or "UNKNOWN_ERROR"
        super().__init__(f"Transition validation failed with errors: {error_codes}")


TRANSITION_SCHEMA = TransitionSchema(
    required_columns=(
        "state",
        "action",
        "reward",
        "next_state",
        "done",
        "episode_id",
        "timestamp",
    ),
    optional_columns=("step_index", "truncated", "info_json", "behavior_log_prob"),
    column_rules={
        "state": ColumnRule(("integer", "floating", "string", "object")),
        "action": ColumnRule(("integer", "floating", "boolean", "string", "object")),
        "reward": ColumnRule(("integer", "floating", "string")),
        "next_state": ColumnRule(("integer", "floating", "string", "object")),
        "done": ColumnRule(("boolean", "integer", "string")),
        "episode_id": ColumnRule(("integer", "floating", "string", "object")),
        "timestamp": ColumnRule(("datetime", "integer", "floating", "string")),
    },
)


def load_transition_dataframe(parquet_path: str | Path) -> pd.DataFrame:
    """Load transition-like data from a parquet file."""
    return pd.read_parquet(Path(parquet_path))


def validate_transition_dataframe(
    df: pd.DataFrame,
    *,
    mode: ValidationMode = "strict",
    schema: TransitionSchema = TRANSITION_SCHEMA,
) -> ValidationResult:
    """Validate transition schema, dtypes, null policy, and timestamp order."""
    if mode not in ("strict", "lenient"):
        raise ValueError("mode must be one of: strict, lenient")

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    if df.empty:
        errors.append(
            ValidationIssue(
                code="EMPTY_DATASET",
                severity="error",
                message="Dataset is empty.",
                remediation="Provide a non-empty transitions dataset.",
            )
        )

    duplicated_mask = pd.Index(df.columns).duplicated(keep=False)
    duplicated_columns = sorted(set(pd.Index(df.columns)[duplicated_mask].tolist()))
    if duplicated_columns:
        errors.append(
            ValidationIssue(
                code="DUPLICATE_COLUMN_NAMES",
                severity="error",
                message="Duplicate column names found.",
                columns=duplicated_columns,
                remediation="Ensure each transition column name is unique.",
            )
        )

    missing_required = [column for column in schema.required_columns if column not in df.columns]
    if missing_required:
        errors.append(
            ValidationIssue(
                code="MISSING_REQUIRED_COLUMN",
                severity="error",
                message="Required transition columns are missing.",
                columns=missing_required,
                remediation=(
                    "Add all required columns: state, action, reward, "
                    "next_state, done, episode_id, timestamp."
                ),
            )
        )
        return _finalize_result(df, mode, errors, warnings)

    _validate_dtypes(df, schema, mode, errors, warnings)
    _validate_nulls(df, schema, mode, errors, warnings)
    _validate_done_values(df, mode, errors, warnings)
    _validate_reward_values(df, mode, errors, warnings)
    _validate_timestamp_order(df, mode, errors, warnings)

    return _finalize_result(df, mode, errors, warnings)


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


def format_validation_report(result: ValidationResult, *, max_items: int = 10) -> str:
    lines = [
        f"schema={result.schema_name}",
        f"mode={result.mode}",
        f"passed={result.passed}",
        f"rows={result.summary.get('row_count', 0)} columns={result.summary.get('column_count', 0)}",
        f"errors={result.summary.get('error_count', 0)} warnings={result.summary.get('warning_count', 0)}",
    ]

    if result.errors:
        lines.append("error_details:")
        for issue in result.errors[:max_items]:
            lines.append(_format_issue(issue))

    if result.warnings:
        lines.append("warning_details:")
        for issue in result.warnings[:max_items]:
            lines.append(_format_issue(issue))

    return "\n".join(lines)


def _format_issue(issue: ValidationIssue) -> str:
    columns = ",".join(issue.columns) if issue.columns else "-"
    samples = ",".join(str(idx) for idx in issue.sample_rows[:5]) if issue.sample_rows else "-"
    return (
        f"- [{issue.severity}] {issue.code} columns={columns} row_count={issue.row_count} "
        f"sample_rows={samples} msg={issue.message} remediation={issue.remediation}"
    )


def _finalize_result(
    df: pd.DataFrame,
    mode: ValidationMode,
    errors: list[ValidationIssue],
    warnings: list[ValidationIssue],
) -> ValidationResult:
    summary = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "error_count": int(len(errors)),
        "warning_count": int(len(warnings)),
    }
    return ValidationResult(
        passed=(len(errors) == 0),
        mode=mode,
        schema_name="offline_rl_transition_v1",
        errors=errors,
        warnings=warnings,
        summary=summary,
    )


def _validate_dtypes(
    df: pd.DataFrame,
    schema: TransitionSchema,
    mode: ValidationMode,
    errors: list[ValidationIssue],
    warnings: list[ValidationIssue],
) -> None:
    for column, rule in schema.column_rules.items():
        if column not in df.columns:
            continue

        category = _dtype_category(df[column])
        if category in rule.allowed_dtype_categories:
            continue

        issue = ValidationIssue(
            code="INVALID_DTYPE",
            severity="error" if mode == "strict" else "warning",
            message=(
                f"Column '{column}' has dtype category '{category}', expected one of "
                f"{rule.allowed_dtype_categories}."
            ),
            columns=[column],
            remediation="Cast this column to an allowed dtype.",
        )
        if mode == "strict":
            errors.append(issue)
        else:
            warnings.append(issue)


def _validate_nulls(
    df: pd.DataFrame,
    schema: TransitionSchema,
    mode: ValidationMode,
    errors: list[ValidationIssue],
    warnings: list[ValidationIssue],
) -> None:
    for column in schema.required_columns:
        null_count = int(df[column].isna().sum())
        if null_count == 0:
            continue

        sample_rows = [int(x) for x in df.index[df[column].isna()].tolist()[:10]]
        issue = ValidationIssue(
            code="NULL_VALUES_IN_REQUIRED_COLUMN",
            severity="error" if mode == "strict" else "warning",
            message=f"Column '{column}' contains null values.",
            columns=[column],
            row_count=null_count,
            sample_rows=sample_rows,
            remediation="Drop or impute rows with nulls in required fields.",
        )
        if mode == "strict":
            errors.append(issue)
        else:
            warnings.append(issue)


def _validate_done_values(
    df: pd.DataFrame,
    mode: ValidationMode,
    errors: list[ValidationIssue],
    warnings: list[ValidationIssue],
) -> None:
    values = df["done"].dropna()
    if values.empty:
        return

    category = _dtype_category(values)
    if category == "boolean":
        return

    numeric = pd.to_numeric(values, errors="coerce")
    invalid_mask = numeric.isna() | ~numeric.isin([0, 1])
    invalid_count = int(invalid_mask.sum())
    if invalid_count == 0:
        return

    sample_rows = [int(x) for x in values.index[invalid_mask].tolist()[:10]]
    issue = ValidationIssue(
        code="INVALID_DONE_VALUES",
        severity="error" if mode == "strict" else "warning",
        message="Column 'done' must be bool or numeric 0/1.",
        columns=["done"],
        row_count=invalid_count,
        sample_rows=sample_rows,
        remediation="Normalize done to bool, or integers 0/1.",
    )
    if mode == "strict":
        errors.append(issue)
    else:
        warnings.append(issue)


def _validate_reward_values(
    df: pd.DataFrame,
    mode: ValidationMode,
    errors: list[ValidationIssue],
    warnings: list[ValidationIssue],
) -> None:
    numeric = pd.to_numeric(df["reward"], errors="coerce")
    finite_mask = np.isfinite(numeric.to_numpy(dtype=np.float64, copy=True))
    invalid_mask = ~pd.Series(finite_mask, index=df.index)
    invalid_count = int(invalid_mask.sum())
    if invalid_count == 0:
        return

    sample_rows = [int(x) for x in df.index[invalid_mask].tolist()[:10]]
    issue = ValidationIssue(
        code="INVALID_REWARD_VALUES",
        severity="error" if mode == "strict" else "warning",
        message="Column 'reward' must be finite numeric values.",
        columns=["reward"],
        row_count=invalid_count,
        sample_rows=sample_rows,
        remediation="Ensure reward is numeric and finite for every row.",
    )
    if mode == "strict":
        errors.append(issue)
    else:
        warnings.append(issue)


def _validate_timestamp_order(
    df: pd.DataFrame,
    mode: ValidationMode,
    errors: list[ValidationIssue],
    warnings: list[ValidationIssue],
) -> None:
    parsed = _parse_timestamp_series(df["timestamp"])

    parse_fail_mask = parsed.isna()
    parse_fail_count = int(parse_fail_mask.sum())
    if parse_fail_count > 0:
        sample_rows = [int(x) for x in df.index[parse_fail_mask].tolist()[:10]]
        issue = ValidationIssue(
            code="INVALID_TIMESTAMP_VALUES",
            severity="error" if mode == "strict" else "warning",
            message="Timestamp values cannot be parsed consistently.",
            columns=["timestamp"],
            row_count=parse_fail_count,
            sample_rows=sample_rows,
            remediation="Use valid datetime strings or valid epoch numeric values.",
        )
        if mode == "strict":
            errors.append(issue)
        else:
            warnings.append(issue)
        return

    tmp = pd.DataFrame({
        "episode_id": df["episode_id"],
        "timestamp": parsed,
    })
    deltas = tmp.groupby("episode_id", dropna=False)["timestamp"].diff()
    bad_mask = deltas < pd.Timedelta(0)
    bad_count = int(bad_mask.sum())
    if bad_count == 0:
        return

    sample_rows = [int(x) for x in df.index[bad_mask].tolist()[:10]]
    issue = ValidationIssue(
        code="NON_MONOTONIC_TIMESTAMP",
        severity="error" if mode == "strict" else "warning",
        message="Timestamps must be non-decreasing inside each episode_id.",
        columns=["episode_id", "timestamp"],
        row_count=bad_count,
        sample_rows=sample_rows,
        remediation="Sort and repair timestamp ordering within each episode.",
    )
    if mode == "strict":
        errors.append(issue)
    else:
        warnings.append(issue)


def _dtype_category(series: pd.Series) -> str:
    if is_bool_dtype(series):
        return "boolean"
    if is_integer_dtype(series):
        return "integer"
    if is_float_dtype(series):
        return "floating"
    if is_datetime64_any_dtype(series):
        return "datetime"
    if is_string_dtype(series):
        return "string"
    if is_object_dtype(series):
        return "object"
    return str(series.dtype)


def _parse_timestamp_series(series: pd.Series) -> pd.Series:
    if is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=True, errors="coerce")

    if is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        unit = _infer_epoch_unit(numeric)
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")

    return pd.to_datetime(series, utc=True, errors="coerce")


def _infer_epoch_unit(series: pd.Series) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna().abs()
    if clean.empty:
        return "s"

    max_value = float(clean.max())
    if max_value < 1e11:
        return "s"
    if max_value < 1e14:
        return "ms"
    if max_value < 1e17:
        return "us"
    return "ns"


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
        # Use integer array if every value is an integer-valued number.
        rounded = np.rint(numeric.to_numpy(dtype=np.float64))
        if np.allclose(rounded, numeric.to_numpy(dtype=np.float64)):
            return rounded.astype(np.int64)
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
