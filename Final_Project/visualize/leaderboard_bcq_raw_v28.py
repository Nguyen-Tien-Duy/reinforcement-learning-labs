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

# Đảm bảo dùng SPAWN cho CUDA
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("spawn")

def evaluate_policy_raw(ep_list, algo, config, is_oracle=False):
    """Giả lập vật lý ở chế độ RAW MODE cho BCQ/CQL"""
    all_costs = []
    all_misses = []
    
    deadline_penalty = getattr(config, "deadline_penalty", 500)
    
    for ep_df in ep_list:
        # Tắt chuẩn hóa ở environment để AI tự dùng scaler nội bộ
        env = CharityGasEnv(ep_df, config, mins=None, maxs=None)
        
        # Hack tốc độ: Gán trực tiếp actions nếu là Oracle
        if is_oracle:
            env.expert_actions = ep_df["action"].to_numpy().astype(int)
            
        obs, _ = env.reset()
        done = False
        ep_cost = 0.0
        
        while not done:
            if is_oracle:
                action = env.expert_actions[min(env.current_step, len(env.expert_actions)-1)]
            else:
                # Đưa về (1, 11) để dự đoán batch của 1
                # d3rlpy algo.predict tự động xử lý observation_scaler nếu có
                res = algo.predict(obs.reshape(1, -1))
                action = res.item() if hasattr(res, 'item') else res
                
            obs, reward, terminated, truncated, info = env.step(action)
            ep_cost += info.get("cost", 0.0)
            
            done = terminated or truncated
            
        # Cộng penalty nếu trễ hạn (V28 Logic)
        is_miss = info.get("deadline_miss", False)
        if is_miss:
            ep_cost += env.queue_size * deadline_penalty
            
        all_costs.append(ep_cost)
        all_misses.append(1 if is_miss else 0)
        
    return np.mean(all_costs), np.mean(all_misses) * 100

def _worker_eval_model(m_path, ep_list, config):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        # Load d3rlpy model
        algo = d3rlpy.load_learnable(m_path, device=device)
        cost, miss = evaluate_policy_raw(ep_list, algo, config, is_oracle=False)
        return {"name": Path(m_path).name, "cost": cost, "miss": miss, "error": None}
    except Exception as e:
        return {"name": Path(m_path).name, "cost": 0, "miss": 0, "error": str(e)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--data", type=str, default="Final_Project/Data/transitions_discrete_v28.parquet")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    # 1. LOAD SETUP
    print(f"[*] Đang nạp dữ liệu cho BCQ Leaderboard (RAW MODE)...")
    import dataclasses
    config_base = TransitionBuildConfig()
    config = dataclasses.replace(config_base, normalize_state=False)

    df = pd.read_parquet(args.data)
    unique_eps = sorted(df['episode_id'].unique())
    test_ids = unique_eps[int(len(unique_eps)*0.9):]
    ep_list = [d.reset_index(drop=True) for _, d in list(df[df['episode_id'].isin(test_ids[:args.episodes])].groupby('episode_id'))]
    del df

    # 2. RUN BASELINE ORACLE
    print(f"[*] Đang chạy Baseline Oracle...")
    c_o, m_o = evaluate_policy_raw(ep_list, None, config, is_oracle=True)
    print(f"✅ XONG BASELINE: Oracle | Chi phí: {c_o:,.0f} | Trễ: {m_o:.1f}%")
    
    results = [["Oracle (Expert)", f"{c_o:,.0f}", f"{m_o:.1f}%"]]
    expert_cost = c_o

    # 3. COLLECT MODELS
    all_model_paths = []
    if args.models:
        for m_path in args.models:
            p = Path(m_path)
            if p.is_dir():
                all_model_paths.extend(list(p.glob("model_*.d3")))
            else:
                all_model_paths.append(str(p))

    # 4. RUN PARALLEL
    if all_model_paths:
        ram_gb = psutil.virtual_memory().available / 1e9
        max_workers = max(1, min(os.cpu_count() or 1, int(ram_gb // 2.0)))
        
        print(f"[+] Đang đánh giá {len(all_model_paths)} mô hình BCQ ở RAW MODE...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_worker_eval_model, p, ep_list, config) for p in all_model_paths]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res["error"]:
                    print(f"❌ Lỗi {res['name']}: {res['error']}")
                else:
                    eff = (expert_cost / res["cost"] * 100) if res["cost"] > 0 else 0
                    print(f"✅ Finish: {res['name']} | Cost: {res['cost']:,.0f} | Trễ: {res['miss']:.1f}%")
                    results.append([res["name"], f"{res['cost']:,.0f}", f"{res['miss']:.1f}%"])

    # 5. PRINT SUMMARY
    print("\n" + "="*60)
    print("🏆 BẢNG VÀNG BCQ (RAW MODE - V28)")
    print("="*60)
    print(tabulate(results, headers=["Model", "True Cost ↓", "Miss Rate ↓"], tablefmt="github"))
    print("="*60)

if __name__ == "__main__":
    main()
