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
    deadline_penalty: float = 2000000.0 # 2.0 points (was 100.0)
    queue_penalty: float = 0.0
    execute_penalty: float = 0.0
    
    # Economics & Urgency
    gas_reference_window: int = 128
    normalize_state: bool = False
    gas_used_col: str = "gas_used"
    gas_limit_col: str = "gas_limit"
    transaction_count_col: str = "transaction_count"
    urgency_alpha: float = 3.0
    urgency_beta: float = 100.0 # 0.0001 per tx (was 0.01)
    reward_scale: float = 1e6
    reward_scale_g: float = 1.0  # Kept for backward compatibility

    # Continuous Action Economy Setup
    C_base: float = 21000.0
    C_mar: float = 15000.0
    gas_to_gwei_scale: float = 1e9  # Divisor to map raw gas (wei) to Gwei unit
    execution_capacity: float = 500.0 # Maximum transactions per block
