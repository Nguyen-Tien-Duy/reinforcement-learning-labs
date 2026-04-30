import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import d3rlpy
import torch
from tabulate import tabulate

# Thêm đường dẫn tới thư mục code để Python tìm thấy module utils
sys.path.append(os.path.abspath("Final_Project/code"))

from utils.offline_rl.enviroment import CharityGasEnv
from utils.offline_rl.config import TransitionBuildConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Đường dẫn tới model")
    parser.add_argument("--data", type=str, default="Final_Project/Data/transitions_v32_C.parquet")
    parser.add_argument("--norm", type=str, default="Final_Project/Data/state_norm_params.json")
    args = parser.parse_args()

    print(f"[*] Đang nạp dữ liệu Test...")
    config = TransitionBuildConfig()
    with open(args.norm, "r") as f:
        norm_data = json.load(f)
        mins, maxs = np.array(norm_data["mins"]), np.array(norm_data["maxs"])

    df = pd.read_parquet(args.data)
    
    # Khôi phục RAW
    for i, col in enumerate(["s_queue", "s_base_fee"]): 
        if col in df.columns and df[col].max() <= 1.01:
            print(f"[!] Dữ liệu đang ở dạng chuẩn hóa. Tiến hành khôi phục RAW...")
            from utils.offline_rl.schema import STATE_COLS
            for j, c in enumerate(STATE_COLS):
                if c in df.columns:
                    df[c] = df[c] * (maxs[j] - mins[j]) + mins[j]
            break

    unique_eps = sorted(df['episode_id'].unique())
    test_ids = unique_eps[int(len(unique_eps)*0.8):]
    
    ep_list = [d.reset_index(drop=True) for _, d in list(df[df['episode_id'].isin(test_ids)].groupby('episode_id'))]
    del df 

    print(f"[*] Đang load model: {args.model}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = d3rlpy.load_learnable(args.model, device=device)

    print(f"[*] Bắt đầu săn tìm Episode bị Miss Deadline...")

    for ep_idx, ep_df in enumerate(ep_list):
        env = CharityGasEnv(ep_df, config, mins=None, maxs=None)
        expert_actions = ep_df["action"].to_numpy().astype(int)
        
        obs, _ = env.reset()
        done = False
        
        trajectory = []
        
        while not done:
            step_idx = env.current_step
            # Oracle action
            expert_a = expert_actions[min(step_idx, len(expert_actions)-1)]
            
            # AI action
            obs_contiguous = np.ascontiguousarray(obs.reshape(1, -1), dtype=np.float32)
            res = model.predict(obs_contiguous)
            ai_a = int(res[0] if isinstance(res, (list, np.ndarray)) else res)
            
            # LẤY GIÁ TRỊ VẬT LÝ TRỰC TIẾP TỪ ENVIRONMENT (Thay vì đọc mảng Obs bị Log-Scale)
            base_fee = env.gas_t_gwei[step_idx]
            queue_size = env.queue_size
            
            # Step environment using AI action
            obs, reward, terminated, truncated, info = env.step(ai_a)
            
            trajectory.append([
                step_idx,
                f"{base_fee:.1f}",
                f"{queue_size:.1f}",
                ai_a,
                expert_a,
                f"{info.get('cost', 0.0):.1f}"
            ])
            
            if terminated or truncated:
                ep_miss = info.get("deadline_miss", False)
                if ep_miss:
                    print(f"\n=======================================================")
                    print(f"🚨 TÌM THẤY EPISODE LỖI (Trễ Deadline)!")
                    print(f"=======================================================")
                    print(f"Episode ID: {ep_df['episode_id'].iloc[0]} (Index: {ep_idx})")
                    print(f"Hàng tồn đọng còn lại: {env.queue_size:.1f}")
                    print(f"Penalty áp dụng: {env.queue_size * config.deadline_penalty:,.0f} Gwei\n")
                    
                    headers = ["Step", "Base Fee (Gwei)", "Queue Size", "AI Action", "Oracle Action", "Step Cost (Gwei)"]
                    print(tabulate(trajectory, headers=headers, tablefmt="github"))
                    
                    print(f"\nPhân tích nhanh cho EPISODE: {ep_df['episode_id'].iloc[0]} (Index: {ep_idx})")
                    print(f"- Penalty áp dụng do Miss: {env.queue_size * config.deadline_penalty:,.0f} Gwei")
                    print(f"- Hàng tồn đọng còn lại: {env.queue_size:.1f}")
                    print(f"- Base Fee cao nhất trong episode: {max([float(x[1]) for x in trajectory]):.1f}")
                    print(f"- Số lần AI chọn Wait (A=0): {sum(1 for x in trajectory if x[3] == 0)}")
                    print(f"- Số lần Oracle chọn Wait (A=0): {sum(1 for x in trajectory if x[4] == 0)}")
                    
                    return # Thoát luôn khi tìm thấy lỗi đầu tiên
                done = True

    print("\n✅ Tuyệt vời! Không tìm thấy bất kỳ tập nào bị Miss Deadline trong lượt test này.")

if __name__ == "__main__":
    main()
