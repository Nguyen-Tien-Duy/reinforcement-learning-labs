import sys, os
from pathlib import Path
import pandas as pd
import numpy as np

# Load simple-offline components
sys.path.append("/mnt/WindowsD/Reinforcement Learning/labs/Final_Project/code")
from utils.offline_rl.enviroment import CharityGasEnv
from utils.offline_rl.config import TransitionBuildConfig
from utils.load_data import build_d3rlpy_dataset
import d3rlpy

# Load data
df = pd.read_parquet("/mnt/WindowsD/Reinforcement Learning/labs/Final_Project/Data/transitions_proxy_fixed.parquet")

# Holdout split logic from simple_offline
temp = df.copy()
temp["_parsed_ts"] = pd.to_datetime(temp["timestamp"], utc=True, errors="coerce")
temp = temp.sort_values("_parsed_ts", na_position="last")
split_idx = int((1.0 - 0.2) * len(temp))
eval_df = temp.iloc[split_idx:].reset_index(drop=True)

# Load model using fallback
bootstrap_n = min(2048, len(eval_df))
bootstrap_df = eval_df.head(bootstrap_n).copy()
bootstrap_dataset = build_d3rlpy_dataset(bootstrap_df, mode="strict")
algo = d3rlpy.algos.DiscreteCQLConfig().create(device="cpu")
algo.build_with_dataset(bootstrap_dataset)
algo.load_model("/mnt/WindowsD/Reinforcement Learning/labs/Final_Project/output/toy_iql_20260411_185317.d3")

config = TransitionBuildConfig(
    action_col="action",
    queue_col=None,
    history_window=3,
    episode_hours=24,
    action_threshold=0.0,
    deadline_penalty=100.0,
    queue_penalty=0.0,
    execute_penalty=0.0,
    gas_reference_window=128,
    normalize_state=True
)

ep_df = next(iter(eval_df.groupby("episode_id")))[1]
env = CharityGasEnv(episode_df=ep_df, config=config)
state, _ = env.reset()

print("Testing first 50 steps:")
for i in range(50):
    action = int(algo.predict(np.array([state]))[0])
    print(f"Step {i:02d}: raw_q={env.queue_size:5.1f}, time={env.time_to_deadline:4.1f}, action={action}, queue_state_term={state[3]:.4f}")
    state, r, term, trunc, info = env.step(action)
    if term or trunc: 
        print(f"Terminated! queue={env.queue_size}, term={term}, trunc={trunc}")
        break
