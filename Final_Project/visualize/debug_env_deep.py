import pandas as pd
import numpy as np
import torch
import d3rlpy
from pathlib import Path
from utils.offline_rl.enviroment import CharityGasEnv
from utils.offline_rl.config import TransitionBuildConfig
import json

def debug_single_model(model_path, data_path, episode_id=None):
    print(f"[*] Debugging model: {model_path}")
    
    # 1. Load Data
    df = pd.read_parquet(data_path)
    if episode_id is None:
        # Lấy đại 1 episode bị trễ hạn (nếu có nhãn deadline_miss hoặc tự tính)
        episode_id = df["episode_id"].unique()[0]
    
    ep_df = df[df["episode_id"] == episode_id].sort_values("step_index")
    print(f"[*] Episode ID: {episode_id}, Steps: {len(ep_df)}")

    # 2. Setup Env & Config
    config = TransitionBuildConfig(
        history_window=3,
        arrival_scale=0.05,
        deadline_penalty=10000000.0,
        urgency_alpha=3.0,
        urgency_beta=100.0,
        reward_scale=1.0,
        action_col="action",
        execution_capacity=500.0
    )
    env = CharityGasEnv(ep_df, config)
    
    # Load normalization
    norm_path = Path("Final_Project/Data/state_norm_params.json")
    if norm_path.exists():
        with open(norm_path, "r") as f:
            params = json.load(f)
            env.mins = np.array(params["mins"], dtype=np.float32)
            env.maxs = np.array(params["maxs"], dtype=np.float32)
            print("[✓] Loaded normalization params.")

    # 3. Load Model
    algo = d3rlpy.load_learnable(model_path)
    
    # 4. Simulation Loop with Deep Debug
    state, _ = env.reset()
    total_reward = 0
    print("\n" + "="*80)
    print(f"{'Step':<5} | {'Raw Q':<8} | {'Action':<6} | {'Reward':<10} | {'Q-Values'} | {'Full State (Normalized)'}")
    print("-" * 120)
    
    for i in range(len(ep_df)):
        with torch.no_grad():
            q_values = algo.predict_value(np.array([state]), np.array([0,1,2,3,4]))
            
        action = int(algo.predict(np.array([state]))[0])
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # In full state để soi
        state_str = "[" + ", ".join([f"{x:.4f}" for x in state]) + "]"
        q_str = np.array2string(q_values, precision=2, separator=',').replace('\n', '')
        print(f"{i:<5} | {env.queue_size:<8.2f} | {action:<6} | {reward:<10.2e} | {q_str:<45} | {state_str}")
        
        state = next_state
        total_reward += reward
        if terminated or truncated:
            break
            
    print("="*80)
    print(f"Total Reward: {total_reward:.2e}")
    print(f"Final Info: {info}")

if __name__ == "__main__":
    # Tìm model mới nhất
    import glob
    log_dirs = glob.glob("d3rlpy_logs/DiscreteCQL_V6*")
    if not log_dirs:
        print("No log dirs found.")
        exit(1)
    latest_dir = max(log_dirs)
    models = glob.glob(f"{latest_dir}/model_35000.d3") # Test model 35k
    if not models:
        models = glob.glob(f"{latest_dir}/*.d3")
    
    debug_single_model(
        models[0], 
        "Final_Project/Data/transitions_discrete_v21.parquet"
    )
