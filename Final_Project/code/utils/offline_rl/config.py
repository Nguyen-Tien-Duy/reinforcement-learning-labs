from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransitionBuildConfig:
    # --- CANONICAL PHYSICS (Locked Source of Truth) ---
    arrival_scale: float = 0.1          # 10% of real traffic to prevent extreme congestion
    deadline_penalty: float = 500.0     # V24: Certified Smart+ Optimal Penalty
    episode_hours: int = 24             # Fixed 24h window
    history_window: int = 3             # 3-step gas history
    
    # --- ENVIRONMENT CONSTANTS ---
    gas_scaling_factor: float = 10.0    # s_g from SPEC: Efficiency = n * (ref - t) / s_g
    gas_to_gwei_scale: float = 1e9      # Divisor for raw gas (wei) to Gwei
    execution_capacity: float = 500.0   # Max transactions per block
    
    # --- OPTIONAL METADATA & COLUMNS ---
    timestamp_col: str = "timestamp"
    gas_col: str = "base_fee_per_gas"
    action_col: str = "action"
    queue_col: str = "queue_size"
    gas_used_col: str = "gas_used"
    gas_limit_col: str = "gas_limit"
    transaction_count_col: str = "transaction_count"
    
    # --- ECONOMICS & URGENCY ---
    gas_reference_window: int = 128
    normalize_state: bool = False        # Forced TRUE for strict pipeline
    urgency_alpha: float = 3.0          # Exponential curve steepness
    urgency_beta: float = 0.0005          # V25: Certified Smart++ Optimal Urgency
    reward_scale: float = 1.0           # Baseline scale (Alpha handling in Model)

    # --- DISCRETE ACTION SPACE (V6) ---
    n_action_bins: int = 5
    action_bins: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)
