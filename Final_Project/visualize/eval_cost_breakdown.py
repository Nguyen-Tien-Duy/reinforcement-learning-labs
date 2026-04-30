import pandas as pd
import numpy as np
import d3rlpy
import sys
import os

# Đảm bảo import được code của project
sys.path.append(os.path.abspath("Final_Project/code"))
from utils.offline_rl.enviroment import CharityGasEnv
from utils.offline_rl.config import TransitionBuildConfig

def evaluate_breakdown(model_path, data_path):
    print(f"[*] Đang nạp cấu hình và dữ liệu từ {data_path}...")
    config = TransitionBuildConfig()
    df = pd.read_parquet(data_path)
    
    unique_eps = sorted(df['episode_id'].unique())
    # Lấy 20% tập test cuối cùng (tương đương với leaderboard)
    test_ids = unique_eps[int(len(unique_eps)*0.8):]
    
    print(f"[*] Đang nạp model từ {model_path}...")
    model = d3rlpy.load_learnable(model_path)
    
    total_pure_gas = 0.0
    total_penalty = 0.0
    total_episodes = len(test_ids)
    miss_count = 0
    
    print(f"[*] Đang chạy Evaluation chi tiết trên {total_episodes} tập...")
    for idx, ep_id in enumerate(test_ids):
        ep_df = df[df['episode_id'] == ep_id].reset_index(drop=True)
        # Tắt chuẩn hóa trạng thái để test raw
        env = CharityGasEnv(ep_df, config, mins=None, maxs=None)
        obs, _ = env.reset()
        done = False
        
        ep_pure_gas = 0.0
        ep_penalty = 0.0
        ep_miss = False
        
        while not done:
            obs_contiguous = np.ascontiguousarray(obs.reshape(1, -1), dtype=np.float32)
            action = int(model.predict(obs_contiguous)[0])
            obs, reward, terminated, truncated, info = env.step(action)
            ep_pure_gas += info.get("cost", 0.0)
            
            if terminated or truncated:
                ep_miss = info.get("deadline_miss", False)
                if ep_miss and env.queue_size > 0:
                    ep_penalty = env.queue_size * config.deadline_penalty
                done = True
        
        total_pure_gas += ep_pure_gas
        total_penalty += ep_penalty
        if ep_miss:
            miss_count += 1
            
        if (idx + 1) % 50 == 0:
            print(f"  ... Đã chạy xong {idx+1}/{total_episodes} tập.")
            
    avg_pure_gas = total_pure_gas / total_episodes
    avg_penalty = total_penalty / total_episodes
    avg_total = avg_pure_gas + avg_penalty
    miss_rate = (miss_count / total_episodes) * 100
    
    print("\n" + "="*60)
    print("🏆 BÓC TÁCH CHI PHÍ - SỰ THẬT VỀ MODEL 490k")
    print("="*60)
    print(f"🔹 1. Tiền Gas thực tế (Pure Gas Cost) : {avg_pure_gas:,.0f} Gwei")
    print(f"🔹 2. Tiền Phạt ảo (Penalty Cost)      : {avg_penalty:,.0f} Gwei")
    print("-" * 60)
    print(f"💰 TỔNG CHI PHÍ HIỂN THỊ (Gas + Phạt)  : {avg_total:,.0f} Gwei")
    print(f"📉 Tỉ lệ Miss Deadline                 : {miss_rate:.2f}% ({miss_count}/{total_episodes} tập)")
    print("="*60)
    
if __name__ == "__main__":
    model = "d3rlpy_logs/DiscreteCQL_V6_20260429_1119_20260429111912/model_490000.d3"
    data = "Final_Project/Data/transitions_v33_L2_Batching_RAW.parquet"
    evaluate_breakdown(model, data)
