from pathlib import Path

import pandas

base_dir = Path(__file__).resolve().parent
data_dir = base_dir.parent / "Data"

# Read and write using paths relative to this script's folder,
# so the script works regardless of current working directory.
input_path = data_dir / "data_2024-04-10_2026-04-10.csv"
output_path = data_dir / "data_2024-04-10_2026-04-10.parquet"

data = pandas.read_csv(input_path)
data.to_parquet(output_path, engine="fastparquet")
