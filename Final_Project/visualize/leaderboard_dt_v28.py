import pandas as pd
import json
import numpy as np
import d3rlpy
import torch
import argparse
import psutil
import os
import concurrent.futures
import multiprocessing
from pathlib import Path
from utils.offline_rl.enviroment import CharityGasEnv
from utils.offline_rl.config import TransitionBuildConfig
from tabulate import tabulate
from d3rlpy.algos import StatefulTransformerWrapper

# Đảm bảo dùng SPAWN cho CUDA
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("spawn")

def evaluate_policy_dt(ep_list, model, target_return, config):
    """Giả lập vật lý cho Decision Transformer dùng dữ liệu RAW (để AI tự scale)"""
    all_costs = []
    all_misses = []
    
    # Lấy penalty từ config
    deadline_penalty = getattr(config, "deadline_penalty", 500)
    
    for ep_df in ep_list:
        # Quan trọng: Mỗi episode cần một wrapper mới để reset history
        # GreedyTransformerActionSampler tự động lấy argmax từ logits và trả về action index
        wrapped_dt = model.as_stateful_wrapper(
            target_return=target_return,
            action_sampler=d3rlpy.algos.GreedyTransformerActionSampler()
        )
        
        # mins=None, maxs=None để chạy RAW
        env = CharityGasEnv(ep_df, config, mins=None, maxs=None)
        obs, _ = env.reset()
        done = False
        ep_cost = 0.0
        last_reward = 0.0
        
        while not done:
            # DT predict tự động cập nhật history và target_return dựa trên reward của bước trước
            # Đưa obs thô (RAW) trực tiếp vào AI để AI tự dùng scaler nội bộ
            action = wrapped_dt.predict(obs, float(last_reward))
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            ep_cost += info.get("cost", 0.0)
            last_reward = float(reward)
            
            done = terminated or truncated
            
        # Cập nhật True Cost (V28 logic): Cộng hình phạt lỡ hạn vào chi phí thực tế
        is_miss = info.get("deadline_miss", False)
        if is_miss:
            penalty_total = env.queue_size * deadline_penalty
            ep_cost += penalty_total

        all_costs.append(ep_cost)
        all_misses.append(1 if is_miss else 0)
        
    return np.mean(all_costs), np.mean(all_misses) * 100

def _worker_eval_dt(m_path, ep_list, target_return, config):
    import traceback
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        model = d3rlpy.load_learnable(m_path, device=device)
        cost, miss = evaluate_policy_dt(ep_list, model, target_return, config)
        return {"name": Path(m_path).name, "cost": cost, "miss": miss, "error": None}
    except Exception as e:
        traceback.print_exc()
        return {"name": Path(m_path).name, "cost": 0, "miss": 0, "error": str(e)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--data", type=str, default="Final_Project/Data/transitions_discrete_v28.parquet")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--target", type=float, default=-1300.0) # Mục tiêu phần thưởng tổng (Oracle mốc)
    args = parser.parse_args()

    # 1. LOAD SETUP
    print(f"[*] Đang nạp dữ liệu cho Decision Transformer Leaderboard (RAW MODE)...")
    import dataclasses
    config_base = TransitionBuildConfig()
    # Tạo bản sao config với normalize_state=False vì đây là Frozen Dataclass
    config = dataclasses.replace(config_base, normalize_state=False)

    df = pd.read_parquet(args.data)
    unique_eps = sorted(df['episode_id'].unique())
    test_ids = unique_eps[int(len(unique_eps)*0.9):]
    ep_list = [d.reset_index(drop=True) for _, d in list(df[df['episode_id'].isin(test_ids[:args.episodes])].groupby('episode_id'))]
    del df

    results = []
    
    # 2. COLLECT DT MODELS
    all_model_paths = []
    if args.models:
        for m_path in args.models:
            p = Path(m_path)
            if p.is_dir():
                all_model_paths.extend(list(p.glob("model_*.d3")))
            else:
                all_model_paths.append(str(p))

    # 3. RUN PARALLEL
    if all_model_paths:
        ram_gb = psutil.virtual_memory().available / 1e9
        # DT nặng hơn MLP, cần nhiều RAM hơn (~3GB/worker)
        max_workers = max(1, min(os.cpu_count() or 1, int(ram_gb // 3.0)))
        
        print(f"[+] Đang đánh giá {len(all_model_paths)} mô hình DT với Target Return = {args.target}...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_worker_eval_dt, str(p), ep_list, args.target, config) for p in all_model_paths]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res["error"]:
                    print(f"❌ Lỗi {res['name']}: {res['error']}")
                else:
                    print(f"✅ DT Finish: {res['name']} | Cost: {res['cost']:,.0f} | Miss: {res['miss']:.1f}%")
                    results.append([res["name"], f"{res['cost']:,.0f}", f"{res['miss']:.1f}%"])

    # 4. PRINT SUMMARY
    print("\n" + "="*60)
    print("🏆 BẢNG VÀNG DECISION TRANSFORMER (RAW MODE - V28)")
    print("="*60)
    print(tabulate(results, headers=["Model", "True Cost ↓", "Miss Rate ↓"], tablefmt="github"))
    print("="*60)

if __name__ == "__main__":
    main()
