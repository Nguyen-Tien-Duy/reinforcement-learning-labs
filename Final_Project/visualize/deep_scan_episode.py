import pandas as pd
import numpy as np
import torch
import d3rlpy
import json
from pathlib import Path
import sys
import os

# Tự động thêm đường dẫn để tìm thấy module utils
sys.path.append(os.path.abspath("Final_Project/code"))

from utils.offline_rl.enviroment import CharityGasEnv
from utils.offline_rl.config import TransitionBuildConfig

def debug_episode(model_path, data_path, episode_id):
    df = pd.read_parquet(data_path)
    ep_df = df[df["episode_id"] == episode_id].sort_values("timestamp").reset_index(drop=True)
    
    print(f"[*] Debugging Episode: {episode_id} | Model: {model_path}")
    
    # Setup Env
    config = TransitionBuildConfig()
    # Physical mode like leaderboard
    import dataclasses
    config = dataclasses.replace(config, normalize_state=False)
    
    # Load normalization params for manual scaling check if needed
    data_dir = Path(data_path).parent
    with open(data_dir / "state_norm_params.json", "r") as f:
        params = json.load(f)
    mins_phys = np.array(params["mins"])
    max_maxs = np.array(params["maxs"])
    
    # Restore physical data for env
    from utils.offline_rl.schema import STATE_COLS
    for i, col in enumerate(STATE_COLS):
        if col in ep_df.columns:
            ep_df[col] = ep_df[col] * (max_maxs[i] - mins_phys[i]) + mins_phys[i]

    env = CharityGasEnv(ep_df, config)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    algo = d3rlpy.load_learnable(model_path, device=device)
    
    obs, _ = env.reset()
    done = False
    
    print("\n" + "="*120)
    print(f"{'Step':<5} | {'Queue':<8} | {'T_Dead':<8} | {'Action':<6} | {'Reward':<10} | {'Q-Values (0-4)'}")
    print("-" * 120)
    
    step = 0
    while not done:
        with torch.no_grad():
            # Get Q-values for all actions
            # d3rlpy v2.x DiscreteCQL: algo.predict_value returns Q-values for (obs, action)
            q_values = []
            for a in range(5):
                q = algo.predict_value(obs.reshape(1, -1), np.array([a]))
                q_values.append(q.item())
            
            action = int(algo.predict(obs.reshape(1, -1))[0])
            
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        q_str = " | ".join([f"{q:7.2f}" for q in q_values])
        print(f"{step:<5} | {env.queue_size:<8.1f} | {env.time_to_deadline:<8.2f} | {action:<6} | {reward:<10.2e} | {q_str}")
        
        obs = next_obs
        done = terminated or truncated
        step += 1
        
    print("="*120)
    print(f"Final Info: {info}")

if __name__ == "__main__":
    model = "d3rlpy_logs/DiscreteCQL_V6_20260422_2104_20260422210405/model_370000.d3"
    data = "Final_Project/Data/transitions_discrete_v32.parquet"
    episode = 1131
    debug_episode(model, data, episode)
