import pandas as pd
import numpy as np
import d3rlpy
import sys
import os
from pathlib import Path
import argparse

# Add the project code root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.offline_rl.enviroment import CharityGasEnv
from utils.offline_rl.config import TransitionBuildConfig

def load_norm_params(data_dir):
    import json
    path = Path(data_dir) / "state_norm_params.json"
    if not path.exists():
        return None, None
    with open(path, "r") as f:
        stats = json.load(f)
    return np.array(stats["mins"]), np.array(stats["maxs"])

def diagnostic_eval(model_path, data_path):
    print(f"[!] Loading model: {model_path}")
    algo = d3rlpy.load_learnable(model_path, device="cpu")
    
    print(f"[!] Loading data: {data_path}")
    df = pd.read_parquet(data_path)
    
    # Pick one episode that failed (miss rate is 47%, so it's easy to find)
    ep_ids = df["episode_id"].unique()
    ep_df = df[df["episode_id"] == ep_ids[0]]
    
    # V22: Use canonical config from SSoT (Single Source of Truth)
    sim_config = TransitionBuildConfig()
    
    # Load column names from schema
    from utils.offline_rl.schema import STATE_COLS
    
    # Synchronization: Load normalization params for the Env
    mins, maxs = load_norm_params(Path(data_path).parent)
    
    if mins is None or maxs is None:
        print("[!] WARNING: Could not load normalization_stats.json! Env will remain in RAW mode.")
    else:
        print("[+] Success: Normalization parameters loaded.")
    
    # Initialize environment with STRICT normalization from SSoT
    env = CharityGasEnv(episode_df=ep_df, config=sim_config, mins=mins, maxs=maxs)
    state, _ = env.reset()
    
    print("\n" + "="*60)
    print("      PROFESSIONAL RL STATE AUDIT - MISSION CONTROL")
    print("="*60)
    
    action_counts = {0:0, 1:0, 2:0, 3:0, 4:0}
    
    for i in range(5): # Audit first 5 steps in depth
        print(f"\n[STEP {i:02d}] Decision Point")
        print("-" * 30)
        
        # 1. Show labelled state
        print(f"{'Field Name':<20} | {'Value':<12} | {'Note'}")
        print("-" * 50)
        for idx, col_name in enumerate(STATE_COLS):
            val = state[idx]
            note = ""
            if idx == 8: note = "<-- QUEUE"
            if idx == 9: note = "<-- DEADLINE (HOURS)"
            if val > 1.05 or val < -0.05: note += " [!! RAW VALUE DETECTED !!]"
            
            print(f"{col_name:<20} | {val:>12.6f} | {note}")
            
        # 2. Predict
        action = int(algo.predict(np.array([state]))[0])
        action_counts[action] += 1
        q_values = algo.predict_value(np.array([state]), np.array([[action]]))
        
        print(f"\n>> AI Decision: ACTION {action} (Ratio {sim_config.action_bins[action]})")
        print(f">> Confidence (Q-Value): {q_values[0]:.4f}")
        
        # 3. Environment Step
        state, reward, term, trunc, info = env.step(action)
        if term or trunc:
            print("\n[!] Episode Terminated.")
            break
            
    print("\n" + "="*60)
    print("\nAction Distribution Summary:", action_counts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()
    diagnostic_eval(args.model, args.data)
