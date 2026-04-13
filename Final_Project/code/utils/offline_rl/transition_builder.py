from __future__ import annotations

from pathlib import Path

import pandas as pd

from .build_reward_episode import build_reward_episode_frame
from .build_state_action import build_state_action_frame
from .config import TransitionBuildConfig


def build_transitions(raw_df: pd.DataFrame, config: TransitionBuildConfig) -> pd.DataFrame:
    state_action_df = build_state_action_frame(raw_df, config)
    return build_reward_episode_frame(state_action_df, config)


def build_transitions_from_parquet(
    raw_parquet_path: str | Path,
    config: TransitionBuildConfig,
    output_path: str | Path | None = None,
    use_oracle: bool = False,
    oracle_ratio: float = 0.5,
    suboptimal_ratio: float = 0.2,
) -> pd.DataFrame:
    raw_df = pd.read_parquet(Path(raw_parquet_path))
    transitions = build_transitions(raw_df, config)
    
    if use_oracle:
        print(f"[+] Applying Hindsight Oracle (Oracle={oracle_ratio}, Suboptimal={suboptimal_ratio})...")
        from .oracle_builder import apply_oracle_to_episodes
        transitions = apply_oracle_to_episodes(
            transitions, 
            config, 
            oracle_ratio=oracle_ratio, 
            suboptimal_ratio=suboptimal_ratio
        )
        
    if output_path is not None:
        transitions.to_parquet(Path(output_path), index=False)
    return transitions
