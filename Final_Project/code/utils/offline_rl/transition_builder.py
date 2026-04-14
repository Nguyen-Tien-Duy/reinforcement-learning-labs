from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .build_reward_episode import build_reward_episode_frame
from .build_state_action import build_state_action_frame
from .config import TransitionBuildConfig


def build_transitions(
    raw_df: pd.DataFrame, 
    config: TransitionBuildConfig,
    use_oracle: bool = False,
    oracle_ratio: float = 0.5,
    suboptimal_ratio: float = 0.2
) -> pd.DataFrame:
    # 1. Build initial state-action frame
    df = build_state_action_frame(raw_df, config)
    
    # 2. Apply Hindsight Oracle if requested
    # We do this BEFORE reward calculation so rewards and next_states reflect oracle decisions
    if use_oracle:
        print(f"[+] Applying Hindsight Oracle (Oracle={oracle_ratio}, Suboptimal={suboptimal_ratio})...")
        from .oracle_builder import apply_oracle_to_episodes
        df = apply_oracle_to_episodes(
            df, 
            config, 
            oracle_ratio=oracle_ratio, 
            suboptimal_ratio=suboptimal_ratio
        )
        # CRITICAL: Recalculate queue dynamics to match new Oracle actions.
        # Without this step, queue_size and state vector still reflect the old
        # proxy action, causing State↔Action mismatch and Policy Collapse.
        from .build_state_action import recalculate_queue_and_state
        df = recalculate_queue_and_state(df, config)
        
    # 3. Calculate rewards and episode transitions (done, next_state)
    return build_reward_episode_frame(df, config)
    
def build_transitions_from_parquet(
    raw_parquet_path: str | Path,
    config: TransitionBuildConfig,
    output_path: str | Path | None = None,
    use_oracle: bool = False,
    oracle_ratio: float = 0.5,
    suboptimal_ratio: float = 0.2,
    config_hash: str | None = None,
) -> pd.DataFrame:
    raw_df = pd.read_parquet(Path(raw_parquet_path))
    transitions = build_transitions(
        raw_df, 
        config,
        use_oracle=use_oracle,
        oracle_ratio=oracle_ratio,
        suboptimal_ratio=suboptimal_ratio
    )
        
    if output_path is not None:
        # Convert to pyarrow table to include custom metadata
        table = pa.Table.from_pandas(transitions, preserve_index=False)
        
        custom_metadata = table.schema.metadata or {}
        if config_hash:
            custom_metadata[b"config_fingerprint"] = config_hash.encode("utf-8")
        
        table = table.replace_schema_metadata(custom_metadata)
        pq.write_table(table, Path(output_path))
        
    return transitions
