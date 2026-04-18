import time
import pandas as pd
import numpy as np
import d3rlpy
import torch
from utils.offline_rl.enviroment import CharityGasEnv
from utils.offline_rl.config import TransitionBuildConfig

df = pd.read_parquet("Final_Project/Data/transitions_discrete_v28.parquet")
unique_eps = sorted(df['episode_id'].unique())
test_ids = unique_eps[int(len(unique_eps)*0.9):]
ep_list = [d.reset_index(drop=True) for _, d in list(df[df['episode_id'].isin(test_ids[:10])].groupby('episode_id'))]

import json
with open("Final_Project/Data/state_norm_params.json", "r") as f:
    norm_data = json.load(f)
    mins, maxs = np.array(norm_data["mins"]), np.array(norm_data["maxs"])

config = TransitionBuildConfig()
model = d3rlpy.load_learnable("d3rlpy_logs/DiscreteBCQ_V6_20260418_0047_20260418004757/model_500000.d3", device="cpu")

t0 = time.time()
env = CharityGasEnv(ep_list[0], config, mins=mins, maxs=maxs)
obs, _ = env.reset()
done = False
steps = 0
while not done:
    action = model.predict(obs.reshape(1, -1)).item()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    steps += 1
t1 = time.time()
print(f"1 Episode ({steps} steps) took {t1-t0:.4f} seconds")
