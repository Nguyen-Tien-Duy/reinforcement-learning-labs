from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

from .schema import (
    METADATA_CONFIG_FINGERPRINT_KEY,
    METADATA_REWARD_MEDIAN_KEY,
    METADATA_REWARD_IQR_KEY
)

def load_transition_dataframe(parquet_path: str | Path) -> pd.DataFrame:
    """Load transition-like data from a parquet file."""
    return pd.read_parquet(Path(parquet_path))

def save_transition_dataframe(
    df: pd.DataFrame, 
    path: str | Path, 
    *, 
    config_fingerprint: str | None = None,
    compute_reward_stats: bool = True
) -> None:
    """
    Save dataframe to parquet and embed metadata for reproducibility and scaling.
    """
    table = pa.Table.from_pandas(df)
    custom_metadata = {}

    if config_fingerprint:
        custom_metadata[METADATA_CONFIG_FINGERPRINT_KEY] = config_fingerprint.encode()
    
    if compute_reward_stats and "reward" in df.columns:
        rewards = pd.to_numeric(df["reward"], errors="coerce").dropna()
        if not rewards.empty:
            median = float(rewards.median())
            q75, q25 = np.percentile(rewards, [75, 25])
            iqr = float(q75 - q25)
            # Avoid divide by zero if rewards are all same
            if iqr == 0: iqr = 1.0
            
            custom_metadata[METADATA_REWARD_MEDIAN_KEY] = str(median).encode()
            custom_metadata[METADATA_REWARD_IQR_KEY] = str(iqr).encode()
    
    # Merge with existing pandas metadata
    existing_meta = table.schema.metadata or {}
    combined_meta = {**existing_meta, **custom_metadata}
    table = table.replace_schema_metadata(combined_meta)
    
    pq.write_table(table, str(path))

def get_reward_stats(parquet_path: str | Path) -> tuple[float, float]:
    """Retrieve Median and IQR from parquet metadata. Defaults to (0.0, 1.0) if not found."""
    pf = pq.read_table(str(parquet_path))
    meta = pf.schema.metadata or {}
    
    median_b = meta.get(METADATA_REWARD_MEDIAN_KEY)
    iqr_b = meta.get(METADATA_REWARD_IQR_KEY)
    
    median = float(median_b.decode()) if median_b else 0.0
    iqr = float(iqr_b.decode()) if iqr_b else 1.0
    
    return median, iqr
