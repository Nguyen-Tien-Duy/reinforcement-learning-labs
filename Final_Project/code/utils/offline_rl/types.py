from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

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
