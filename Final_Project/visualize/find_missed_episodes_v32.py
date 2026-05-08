import pandas as pd
import json
import numpy as np
import d3rlpy
import torch
import argparse
import os
from pathlib import Path
import sys

# Tự động thêm đường dẫn để tìm thấy module utils
sys.path.append(os.path.abspath("Final_Project/code"))

from utils.offline_rl.enviroment import CharityGasEnv
from utils.offline_rl.config import TransitionBuildConfig

def find_missed_episodes(ep_list, algo, config):
    missed_ids = []
    
    # Use the penalty from config if available, or default to what was in the training log
    deadline_penalty = getattr(config, "deadline_penalty", 20000)
    
    for ep_df in ep_list:
        ep_id = ep_df["episode_id"].iloc[0]
        env = CharityGasEnv(ep_df, config, mins=None, maxs=None)
        
        obs, _ = env.reset()
        done = False
        
        while not done:
            with torch.no_grad():
                res = algo.predict(obs.reshape(1, -1))
            action = res.item() if hasattr(res, 'item') else res
                
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
        is_miss = info.get("deadline_miss", False)
        if is_miss:
            missed_ids.append(ep_id)
            
    return missed_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, default="Final_Project/Data/transitions_discrete_v32.parquet")
    parser.add_argument("--episodes", type=int, default=241)
    args = parser.parse_args()

    print(f"[*] Đang tìm episodes bị Miss bởi model: {args.model}")
    import dataclasses
    config_base = TransitionBuildConfig()
    # Ensure physical mode logic matches leaderboard
    config = dataclasses.replace(config_base, normalize_state=False)

    df = pd.read_parquet(args.data)
    
    # Inverse Normalization (same as leaderboard)
    try:
        data_dir = Path(args.data).parent
        with open(data_dir / "state_norm_params.json", "r") as f:
            params = json.load(f)
        mins_phys = np.array(params["mins"])
        max_maxs = np.array(params["maxs"])
        
        from utils.offline_rl.schema import STATE_COLS
        for i, col in enumerate(STATE_COLS):
            if col in df.columns:
                df[col] = df[col] * (max_maxs[i] - mins_phys[i]) + mins_phys[i]
    except Exception as e:
        print(f"⚠️ Cảnh báo: Không thể phục hồi vật lý ({e})")

    unique_eps = sorted(df['episode_id'].unique())
    # Match the 10% test split from leaderboard (last 10% of data)
    test_ids_pool = unique_eps[int(len(unique_eps)*0.9):]
    ep_list = [d.reset_index(drop=True) for _, d in list(df[df['episode_id'].isin(test_ids_pool[:args.episodes])].groupby('episode_id'))]
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    algo = d3rlpy.load_learnable(args.model, device=device)
    
    missed_ids = find_missed_episodes(ep_list, algo, config)
    
    print("\n" + "="*50)
    print(f"KẾT QUẢ KIỂM TRA:")
    print(f"Tổng số episodes đánh giá: {len(ep_list)}")
    print(f"Số lượng bị Miss: {len(missed_ids)}")
    if missed_ids:
        print(f"Danh sách Episode ID bị Miss: {missed_ids}")
    else:
        print("🎉 Không tìm thấy episode nào bị Miss trong tập này!")
    print("="*50)

if __name__ == "__main__":
    main()
