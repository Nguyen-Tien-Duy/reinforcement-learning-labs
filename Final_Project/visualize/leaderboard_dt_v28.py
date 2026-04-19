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
import sys

# Tự động thêm đường dẫn để tìm thấy module utils
sys.path.append(os.path.abspath("Final_Project/code"))

from utils.offline_rl.enviroment import CharityGasEnv
from utils.offline_rl.config import TransitionBuildConfig
from tabulate import tabulate
from d3rlpy.algos import StatefulTransformerWrapper
from tqdm import tqdm

# Cấu hình LOGGING để đỡ rác màn hình
import logging
logging.getLogger("d3rlpy").setLevel(logging.ERROR)

def log_transform_reward(r: float) -> float:
    """Khớp chính xác với phép biến đổi trong train_dt_cloud.py và simple-offline.py"""
    return float(np.sign(r) * np.log1p(np.abs(r)))

# Đảm bảo dùng SPAWN cho CUDA
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("spawn")

def _run_single_episode(ep_df, model, target_return, config):
    """Hàm worker để chạy 1 episode đơn lẻ (Dùng cho song song hóa)"""
    # Mỗi worker cần bộ nén và history riêng biệt để không bị loạn
    wrapped_dt = model.as_stateful_wrapper(
        target_return=target_return,
        action_sampler=d3rlpy.algos.GreedyTransformerActionSampler()
    )
    
    env = CharityGasEnv(ep_df, config, mins=None, maxs=None)
    obs, _ = env.reset()
    done = False
    ep_cost = 0.0
    last_reward = 0.0
    deadline_penalty = getattr(config, "deadline_penalty", 500)

    while not done:
        # AI nhận reward đã được biến đổi Log
        action = wrapped_dt.predict(obs, log_transform_reward(last_reward))
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        ep_cost += info.get("cost", 0.0)
        last_reward = float(reward)
        done = terminated or truncated
        
    is_miss = info.get("deadline_miss", False)
    if is_miss:
        ep_cost += env.queue_size * deadline_penalty
        
    return ep_cost, (1 if is_miss else 0)

def evaluate_policy_dt(ep_list, model, target_return, config):
    """Giả lập vật lý song song hóa toàn bộ CPU cho DT"""
    all_costs = []
    all_misses = []
    
    # Số lượng worker tối ưu (vắt kiệt CPU)
    max_workers = os.cpu_count() or 1
    
    print(f"[*] Đang vắt kiệt {max_workers} luồng CPU để giả lập {len(ep_list)} tập đoàn...")
    
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit toàn bộ episodes vào hàng chờ xử lý
        futures = [executor.submit(_run_single_episode, ep, model, target_return, config) for ep in ep_list]
        
        # tqdm hiển thị thanh tiến trình nhảy theo thời gian thực
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(ep_list), desc="Simulating DT"):
            cost, miss = f.result()
            all_costs.append(cost)
            all_misses.append(miss)
        
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
    parser.add_argument("--target", type=float, default=460.0) # Mốc cao thủ (Top 10% Oracle)
    args = parser.parse_args()

    # 1. LOAD SETUP
    print(f"[*] Đang nạp dữ liệu cho Decision Transformer Leaderboard (RAW MODE)...")
    import dataclasses
    config_base = TransitionBuildConfig()
    # Tạo bản sao config với normalize_state=False vì đây là Frozen Dataclass
    config = dataclasses.replace(config_base, normalize_state=False)

    df = pd.read_parquet(args.data)
    
    # === INVERSE NORMALIZATION (Khôi phục vật lý cho Leaderboard - BẮT BUỘC cho RAW MODE) ===
    try:
        data_dir = Path(args.data).parent
        with open(data_dir / "state_norm_params.json", "r") as f:
            params = json.load(f)
        mins_phys = np.array(params["mins"])
        max_maxs = np.array(params["maxs"])
        
        from utils.offline_rl.schema import STATE_COLS
        print(f"[*] Phục hồi vật lý cho dữ liệu đánh giá DT...")
        for i, col in enumerate(STATE_COLS):
            if col in df.columns:
                df[col] = df[col] * (max_maxs[i] - mins_phys[i]) + mins_phys[i]
        print("✅ Khôi phục RAW DATA thành công.")
    except Exception as e:
        print(f"⚠️ Cảnh báo: Không thể phục hồi vật lý ({e}).")

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

    # 3. RUN EVAL
    if all_model_paths:
        for p in all_model_paths:
            print(f"\n[+] ĐANG ĐÁNH GIÁ: {Path(p).name}")
            res = _worker_eval_dt(str(p), ep_list, args.target, config)
            if res["error"]:
                print(f"❌ Lỗi {res['name']}: {res['error']}")
            else:
                results.append([res["name"], f"{res['cost']:,.0f}", f"{res['miss']:.1f}%"])

    # 4. PRINT SUMMARY
    print("\n" + "="*60)
    print("🏆 BẢNG VÀNG DECISION TRANSFORMER (RAW MODE - V28)")
    print("="*60)
    print(tabulate(results, headers=["Model", "True Cost ↓", "Miss Rate ↓"], tablefmt="github"))
    print("="*60)

if __name__ == "__main__":
    main()
