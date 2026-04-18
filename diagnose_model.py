import pandas as pd
import json
import numpy as np
import d3rlpy
import torch
from pathlib import Path
from utils.offline_rl.enviroment import CharityGasEnv
from utils.offline_rl.config import TransitionBuildConfig

def diagnose(model_path, data_path, norm_path, n_episodes=5):
    # 1. Load Setup
    config = TransitionBuildConfig()
    with open(norm_path, "r") as f:
        norm_data = json.load(f)
        mins, maxs = np.array(norm_data["mins"]), np.array(norm_data["maxs"])

    # 2. Load Model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = d3rlpy.load_learnable(model_path, device=device)

    # 3. Load Data
    df = pd.read_parquet(data_path)
    unique_eps = sorted(df['episode_id'].unique())
    test_ids = unique_eps[int(len(unique_eps)*0.9):]
    ep_list = [d.reset_index(drop=True) for _, d in list(df[df['episode_id'].isin(test_ids[:n_episodes])].groupby('episode_id'))]

    print(f"--- Diagnosing Model: {model_path} ---")
    
    all_stats = []
    
    for i, ep_df in enumerate(ep_list):
        env = CharityGasEnv(ep_df, config, mins=mins, maxs=maxs)
        obs, _ = env.reset()
        done = False
        
        history = []
        
        while not done:
            # Predict
            x = obs.reshape(1, -1)
            action = model.predict(x).item()
            
            # Record state before step
            # Features: [0:gas, 8:queue, 9:time_left]
            history.append({
                "step": env.current_step,
                "gas": obs[0],
                "queue": obs[8],
                "time_left": obs[9],
                "action": action,
                "expert_action": ep_df.iloc[env.current_step]["action"] if env.current_step < len(ep_df) else 0
            })
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
        h_df = pd.DataFrame(history)
        print(f"\nEpisode {i+1} (ID: {ep_df['episode_id'].iloc[0]}) Summary:")
        print(f"  Actions chosen: {h_df['action'].value_counts().to_dict()}")
        print(f"  Expert actions: {h_df['expert_action'].value_counts().to_dict()}")
        
        # Correlation check
        corr = h_df[['queue', 'time_left', 'action']].corr()['action']
        print(f"  Correlation Action vs Queue: {corr['queue']:.4f}")
        print(f"  Correlation Action vs TimeLeft: {corr['time_left']:.4f}")
        
        all_stats.append(h_df)

    # Aggregated analysis
    full_h = pd.concat(all_stats)
    print("\n=== GLOBAL ANALYSIS ===")
    print(f"Most frequent action: {full_h['action'].mode()[0]}")
    print("Action distribution across all test steps:")
    dist = full_h['action'].value_counts(normalize=True).sort_index() * 100
    for act, perc in dist.items():
        print(f"  Action {act}: {perc:.1f}%")

if __name__ == "__main__":
    diagnose(
        "d3rlpy_logs/DiscreteBCQ_V6_20260418_0047_20260418004757/model_15000.d3",
        "Final_Project/Data/transitions_discrete_v28.parquet",
        "Final_Project/Data/state_norm_params.json"
    )
