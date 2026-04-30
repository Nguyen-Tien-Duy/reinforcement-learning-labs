import pandas as pd
import numpy as np
import sys
import os
import argparse
from tqdm import tqdm

sys.path.append('Final_Project/code')
from utils.offline_rl.enviroment import CharityGasEnv
from utils.offline_rl.config import TransitionBuildConfig
import d3rlpy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to d3rlpy model (.d3)")
    parser.add_argument("--data", type=str, default="Final_Project/Data/transitions_v32_C.parquet")
    args = parser.parse_args()

    print("[*] Loading data...")
    df = pd.read_parquet(args.data)
    
    # [SOTA FIX] Khôi phục vật lý toàn bộ dataset trước khi tạo Env
    print("[*] Inverse Normalizing dataset to prevent Double Compression...")
    import json
    with open("Final_Project/Data/state_norm_params.json", "r") as f:
        params = json.load(f)
        mins = np.array(params["mins"])
        maxs = np.array(params["maxs"])
    
    from utils.offline_rl.schema import STATE_COLS
    for j, c in enumerate(STATE_COLS):
        if c in df.columns and df[c].max() <= 1.01:
            df[c] = df[c] * (maxs[j] - mins[j]) + mins[j]
    
    unique_episodes = sorted(df["episode_id"].unique())
    # Holdout is the last 20%
    split_idx = int(len(unique_episodes) * 0.8)
    eval_ids = unique_episodes[split_idx:]
    
    print(f"[*] Total eval episodes: {len(eval_ids)}")
    
    print("[*] Loading model...")
    model = d3rlpy.load_learnable(args.model)
    
    config = TransitionBuildConfig()
    
    results = []
    
    for ep_id in tqdm(eval_ids, desc="Evaluating Episodes"):
        ep_df = df[df["episode_id"] == ep_id].reset_index(drop=True)
        
        # --- Evaluate AI ---
        env_ai = CharityGasEnv(ep_df, config, mins=None, maxs=None)
        obs, _ = env_ai.reset()
        done = False
        ai_cost = 0.0
        ai_miss = False
        while not done:
            obs_contiguous = np.ascontiguousarray(obs.reshape(1, -1), dtype=np.float32)
            res = model.predict(obs_contiguous)
            ai_a = int(res[0] if isinstance(res, (list, np.ndarray)) else res)
            obs, reward, term, trunc, info = env_ai.step(ai_a)
            ai_cost += info.get("cost", 0.0)
            done = term or trunc
        ai_miss = info.get("deadline_miss", False)
        
        # --- Evaluate Oracle ---
        env_oracle = CharityGasEnv(ep_df, config, mins=None, maxs=None)
        env_oracle.reset()
        done = False
        oracle_cost = 0.0
        oracle_miss = False
        while not done:
            oracle_a = int(ep_df["action"].iloc[env_oracle.current_step])
            _, _, term, trunc, info = env_oracle.step(oracle_a)
            oracle_cost += info.get("cost", 0.0)
            done = term or trunc
        oracle_miss = info.get("deadline_miss", False)
        
        results.append({
            "episode_id": ep_id,
            "ai_cost": ai_cost,
            "oracle_cost": oracle_cost,
            "ai_miss": ai_miss,
            "oracle_miss": oracle_miss
        })

    res_df = pd.DataFrame(results)
    
    # Filter out missed episodes
    valid_df = res_df[(~res_df["ai_miss"]) & (~res_df["oracle_miss"])]
    missed_df = res_df[res_df["ai_miss"] | res_df["oracle_miss"]]
    
    print("\n" + "="*50)
    print("                KẾT QUẢ ĐỐI ĐẦU 1-1")
    print("="*50)
    print(f"- Số tập hoàn thành (Cả 2 đều ko Miss): {len(valid_df)}")
    print(f"- Số tập lỗi (Miss deadline): {len(missed_df)}")
    if len(missed_df) > 0:
        print("  Danh sách tập Miss:", missed_df["episode_id"].tolist())
        
    ai_avg = valid_df["ai_cost"].mean()
    oracle_avg = valid_df["oracle_cost"].mean()
    
    print("\n--- Trên các tập HOÀN THÀNH ---")
    print(f"💰 Trung bình chi phí AI     : {ai_avg:,.2f} Gwei")
    print(f"💰 Trung bình chi phí Oracle : {oracle_avg:,.2f} Gwei")
    
    savings = oracle_avg - ai_avg
    print(f"\n=> AI tiết kiệm hơn Oracle : {savings:,.2f} Gwei/Episode")
    print(f"=> Hiệu năng AI / Oracle   : {(oracle_avg / ai_avg) * 100:.2f}%")
    
    ai_wins = len(valid_df[valid_df["ai_cost"] < valid_df["oracle_cost"]])
    oracle_wins = len(valid_df[valid_df["ai_cost"] > valid_df["oracle_cost"]])
    ties = len(valid_df[valid_df["ai_cost"] == valid_df["oracle_cost"]])
    
    print(f"\n🏆 Tỷ số chiến thắng: AI thắng {ai_wins} tập | Oracle thắng {oracle_wins} tập | Hòa {ties} tập")

if __name__ == "__main__":
    main()
