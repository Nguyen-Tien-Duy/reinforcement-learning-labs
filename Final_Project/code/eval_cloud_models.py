import argparse
import pandas as pd
import numpy as np
import d3rlpy
import torch
from pathlib import Path
from utils.offline_rl.enviroment import CharityGasEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="Đường dẫn file .d3 đã train từ Cloud")
    parser.add_argument("--data", type=Path, default="Final_Project/Data/transitions_discrete_v27.parquet")
    parser.add_argument("--episodes", type=int, default=20, help="Số vòng chạy thử nghiệm (20 là đủ chuẩn)")
    args = parser.parse_args()

    print(f"[*] Loading data to find Test Set: {args.data}")
    df = pd.read_parquet(args.data)
    
    # Đảm bảo logic cắt 10% cuối y hệt Cloud
    unique_episodes = sorted(df["episode_id"].unique())
    split_idx = int(len(unique_episodes) * 0.9)
    test_ids = unique_episodes[split_idx:]
    test_df = df[df["episode_id"].isin(test_ids)].copy()
    
    print(f"[+] Local Test Set: {len(test_ids)} episodes found (Future Data).")

    # Load Model (Bất kể là BCQ hay Decision Transformer)
    print(f"[*] Loading Model from {args.model}...")
    # Tự động chọn device để tránh văng lỗi nếu máy bác ko có GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Note: Decision Transformer yêu cầu load hơi khác một chút, 
    # nhưng d3rlpy v2 hỗ trợ load tự động qua d3rlpy.load_learnable
    model = d3rlpy.load_learnable(args.model, device=device)

    # 4) Chạy giả lập chấm điểm vật lý
    print(f"[*] Starting Physical Simulation on {args.episodes} episodes...")
    env = CharityGasEnv(test_df)
    
    all_costs = []
    all_misses = []

    for i in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_cost = 0
        
        while not done:
            # Lấy hành động từ não bộ d3rlpy
            action = model.predict([obs])[0]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
        res = env.get_episode_stats()
        all_costs.append(res['total_cost'])
        all_misses.append(1 if res['sla_miss'] else 0)
        print(f"  - Ep {i+1}: Cost={res['total_cost']:.0f} | Miss={res['sla_miss']}")

    print("\n" + "="*30)
    print(f"FINAL REPORT FOR: {args.model.name}")
    print(f"  - Mean Cost: {np.mean(all_costs):,.0f}")
    print(f"  - SLA Miss Rate: {np.mean(all_misses)*100:.2f}%")
    print("="*30)

if __name__ == "__main__":
    main()
