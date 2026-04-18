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

# Thiết lập SPAWN cho Multiprocessing (Để an toàn với CUDA)
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("spawn")

def evaluate_policy(ep_list, policy_func, config, mins, maxs, is_oracle=False):
    """Giả lập vật lý - Đã được tối ưu tốc độ"""
    all_costs = []
    all_misses = []
    
    for ep_df in ep_list:
        env = CharityGasEnv(ep_df, config, mins=mins, maxs=maxs)
        # Hack tốc độ: Gán trực tiếp actions nếu là Oracle
        if is_oracle:
            env.expert_actions = ep_df["action"].to_numpy().astype(int)
            
        obs, _ = env.reset()
        done = False
        ep_cost = 0.0
        ep_miss = False
        
        while not done:
            if is_oracle:
                action = env.expert_actions[min(env.current_step, len(env.expert_actions)-1)]
            else:
                # Đưa về (1, 11) để dự đoán batch của 1
                res = policy_func(obs.reshape(1, -1))
                action = res.item() if hasattr(res, 'item') else res
                
            obs, reward, terminated, truncated, info = env.step(action)
            ep_cost += info.get("cost", 0.0)
            
            if terminated or truncated:
                ep_miss = info.get("deadline_miss", False)
                if ep_miss and env.queue_size > 0:
                    # BẮT BÀI ĐÁNH TRÁO KHÁI NIỆM!
                    # Hàng chưa bán bị thanh lý ở mức phạt 500 Gwei/đơn (config.deadline_penalty)
                    penalty_cost = env.queue_size * config.deadline_penalty
                    ep_cost += penalty_cost
                done = True
            
        all_costs.append(ep_cost)
        all_misses.append(1 if ep_miss else 0)
        
    return np.mean(all_costs), np.mean(all_misses) * 100

def _worker_eval_model(m_path, ep_list, config, mins, maxs):
    """Worker CHỈ load model và chạy, data đã được nạp sẵn"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        model = d3rlpy.load_learnable(m_path, device=device)
        cost, miss = evaluate_policy(ep_list, model.predict, config, mins, maxs)
        return {"name": Path(m_path).name, "cost": cost, "miss": miss, "error": None}
    except Exception as e:
        return {"name": Path(m_path).name, "cost": 0, "miss": 0, "error": str(e)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--data", type=str, default="Final_Project/Data/transitions_discrete_v28.parquet")
    parser.add_argument("--norm", type=str, default="Final_Project/Data/state_norm_params.json")
    parser.add_argument("--episodes", type=int, default=10) # Mặc định 10 cho nhanh
    args = parser.parse_args()

    # 1. NẠP DỮ LIỆU & CẤU HÌNH (MỘT LẦN DUY NHẤT)
    print(f"[*] Nạp dữ liệu Test V28 vào RAM (Chỉ nạp 1 lần)...")
    config = TransitionBuildConfig()
    with open(args.norm, "r") as f:
        norm_data = json.load(f)
        mins, maxs = np.array(norm_data["mins"]), np.array(norm_data["maxs"])

    df = pd.read_parquet(args.data)
    unique_eps = sorted(df['episode_id'].unique())
    test_ids = unique_eps[int(len(unique_eps)*0.9):]
    # Chỉ bốc đúng số lượng episode cần test để tiết kiệm RAM
    ep_list = [d.reset_index(drop=True) for _, d in list(df[df['episode_id'].isin(test_ids[:args.episodes])].groupby('episode_id'))]
    del df # Xóa dataframe gốc để giải phóng RAM

    results = []

    # 2. CHẠY BASELINES
    print("[+] Đang chạy Baselines (Random & Oracle)...")
    c_r, m_r = evaluate_policy(ep_list, lambda x: np.random.randint(0, 5), config, mins, maxs)
    print(f"✅ XONG BASELINE: Random | Chi phí: {c_r:,.0f} | Trễ: {m_r:.1f}%")
    results.append(["Random (Baseline)", f"{c_r:,.0f}", f"{m_r:.1f}%", "-"])

    c_o, m_o = evaluate_policy(ep_list, None, config, mins, maxs, is_oracle=True)
    print(f"✅ XONG BASELINE: Oracle (Expert) | Chi phí: {c_o:,.0f} | Trễ: {m_o:.1f}%")
    results.append(["Oracle (Expert 100%)", f"{c_o:,.0f}", f"{m_o:.1f}%", "100.0%"])
    expert_cost = c_o

    # 3. CHẶN THU THẬP MODEL
    all_model_paths = []
    import re
    def extract_step(path):
        m = re.search(r'model_(\d+)\.d3', str(path))
        return int(m.group(1)) if m else 0
        
    if args.models:
        for m_path in args.models:
            p = Path(m_path)
            if p.is_dir():
                all_model_paths.extend(sorted(list(p.glob("*.d3")), key=extract_step))
            elif p.suffix == ".d3":
                all_model_paths.append(str(p))

    if all_model_paths:
        ram_gb = psutil.virtual_memory().available / 1e9
        # Mỗi worker chỉ nạp 1 model và ngậm 1 phần nhỏ ep_list nên có thể chạy nhiều core hơn
        max_workers = max(1, min(os.cpu_count() or 1, int(ram_gb // 2.0)))
        
        print(f"[+] Tìm thấy {len(all_model_paths)} mô hình. Đang triển khai trên {max_workers} nhân CPU...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Gửi ep_list trực tiếp cho các worker
            futures = [executor.submit(_worker_eval_model, p, ep_list, config, mins, maxs) for p in all_model_paths]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res["error"]:
                    print(f"❌ Lỗi {res['name']}: {res['error']}")
                else:
                    eff = (expert_cost / res["cost"] * 100) if res["cost"] > 0 else 0
                    print(f"✅ Hoàn thành: {res['name']} | Chi phí: {res['cost']:,.0f} | Trễ: {res['miss']:.1f}%")
                    results.append([res["name"], f"{res['cost']:,.0f}", f"{res['miss']:.1f}%", f"{eff:.1f}%"])

    # 4. IN KẾT QUẢ
    headers = ["Thí Sinh", "Chi phí (Gwei) ↓", "Tỉ lệ Trễ (Miss) ↓", "Hiệu năng vs Expert ↑"]
    header_rows = results[:2]
    model_rows = sorted(results[2:], key=lambda x: float(x[3].strip('%')), reverse=True)
    
    print("\n" + "="*80)
    print("🏆 BẢNG VÀNG CÔNG NGHỆ Ethereum RL - PHIÊN BẢN V28 (ULTRA SPEED)")
    print("="*80)
    print(tabulate(header_rows + model_rows, headers=headers, tablefmt="github"))
    print("="*80)

if __name__ == "__main__":
    main()
