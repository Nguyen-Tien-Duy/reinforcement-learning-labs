from __future__ import annotations

from .types import ColumnRule, TransitionSchema

# ─── Canonical 11-dimensional State Column Names ───────────────────────────
# These are the SINGLE SOURCE OF TRUTH for all pipeline modules.
# Index mapping must match the order in build_state_action.py:
#   [s_gas_t0, s_gas_t1, s_gas_t2,       ← Gas History (3 lags)
#    s_congestion, s_momentum, s_accel,   ← Market Microstructure
#    s_surprise, s_backlog,               ← Pseudo-Mempool
#    s_queue, s_time_left, s_gas_ref]     ← Agent Internal State
STATE_COLS = [
    "s_gas_t0",       # 0: Gas price at t
    "s_gas_t1",       # 1: Gas price at t-1
    "s_gas_t2",       # 2: Gas price at t-2
    "s_congestion",   # 3: Block congestion p_t = (gas_used - target) / target
    "s_momentum",     # 4: Log-return of gas price m_t
    "s_accel",        # 5: Acceleration (2nd diff) of gas a_t
    "s_surprise",     # 6: Transaction count Z-score u_t (Pseudo-Mempool)
    "s_backlog",      # 7: EWMA backlog pressure b_t
    "s_queue",        # 8: Agent's pending transaction queue
    "s_time_left",    # 9: Remaining hours to deadline
    "s_gas_ref",      # 10: Rolling mean gas reference (128-block window)
]

NEXT_STATE_COLS = [f"ns_{c[2:]}" if c.startswith("s_") else f"ns_{c}" for c in STATE_COLS]

STATE_IDX = {name: i for i, name in enumerate(STATE_COLS)}
# Convenience accessors
Q_IDX   = STATE_IDX["s_queue"]      # 8
T_IDX   = STATE_IDX["s_time_left"]  # 9

TRANSITION_SCHEMA = TransitionSchema(
    required_columns=(
        *STATE_COLS,
        "action",
        "reward",
        *NEXT_STATE_COLS,
        "done",
        "episode_id",
        "timestamp",
    ),
    optional_columns=("step_index", "truncated", "info_json", "behavior_log_prob"),
    column_rules={
        **{c: ColumnRule(("floating",)) for c in STATE_COLS},
        **{c: ColumnRule(("floating",)) for c in NEXT_STATE_COLS},
        "action":     ColumnRule(("integer", "floating", "boolean", "string", "object")),
        "reward":     ColumnRule(("integer", "floating", "string")),
        "done":       ColumnRule(("boolean", "integer", "string")),
        "episode_id": ColumnRule(("integer", "floating", "string", "object")),
        "timestamp":  ColumnRule(("datetime", "integer", "floating", "string")),
    },
)
