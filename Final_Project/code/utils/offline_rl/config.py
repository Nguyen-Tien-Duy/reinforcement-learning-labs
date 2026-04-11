from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransitionBuildConfig:
    timestamp_col: str = "timestamp"
    gas_col: str = "base_fee_per_gas"
    action_col: str | None = None
    queue_col: str | None = None
    history_window: int = 3
    episode_hours: int = 24
    action_threshold: float = 0.0
    deadline_penalty: float = 100.0
    queue_penalty: float = 0.0
    execute_penalty: float = 0.0
    gas_reference_window: int = 128
    normalize_state: bool = True
