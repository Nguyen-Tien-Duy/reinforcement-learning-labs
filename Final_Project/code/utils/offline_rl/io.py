from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_transition_dataframe(parquet_path: str | Path) -> pd.DataFrame:
    """Load transition-like data from a parquet file."""
    return pd.read_parquet(Path(parquet_path))
