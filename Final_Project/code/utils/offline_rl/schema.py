from __future__ import annotations

from .types import ColumnRule, TransitionSchema

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
