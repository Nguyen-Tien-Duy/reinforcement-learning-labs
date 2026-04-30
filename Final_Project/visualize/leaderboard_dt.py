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
import os

# [PATH FIX] Thêm đường dẫn tới thư mục code để Python tìm thấy module utils
sys.path.append(os.path.abspath("Final_Project/code"))

from utils.offline_rl.enviroment import CharityGasEnv
from utils.offline_rl.config import TransitionBuildConfig
from tabulate import tabulate

# Thiết lập SPAWN cho Multiprocessing (Để an toàn với CUDA)
if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("spawn")

def evaluate_policy(ep_list, model, config, mins, maxs, is_oracle=False, model_name="", target_return=475.0):
    """Giả lập vật lý - Đã được tối ưu tốc độ"""
    all_costs = []
    all_pure_gas = []
    all_penalties = []
    all_misses = []
    
    tot_exec_txs = 0
    tot_miss_txs = 0
    
    for i, ep_df in enumerate(ep_list):
        # [SOTA FIX] Tắt chuẩn hóa ở environment để tránh Nén Kép (Double Compression)
        # Vì model d3rlpy đã có Observation Scaler nội bộ
        env = CharityGasEnv(ep_df, config, mins=None, maxs=None)
        
        # Hack tốc độ: Gán trực tiếp actions nếu là Oracle
        if is_oracle:
            env.expert_actions = ep_df["action"].to_numpy().astype(int)
            
        is_dt = "DecisionTransformer" in str(type(model)) if model else False
        
        if is_dt:
            # Sử dụng Wrapper chính chủ của d3rlpy thay vì tự build tensor
            wrapper = model.as_stateful_wrapper(target_return=target_return)

        # [SOTA SPEED] Memory Alignment cho Inference
        obs, _ = env.reset()
        done = False
        ep_cost = 0.0
        ep_miss = False
        penalty_cost = 0.0
        reward = 0.0 # Reward khởi tạo cho step đầu tiên

        while not done:
            if is_oracle:
                action = env.expert_actions[min(env.current_step, len(env.expert_actions)-1)]
            elif is_dt:
                # Log transform reward giống hệt lúc train trước khi đưa vào wrapper
                transformed_rew = np.sign(reward) * np.log1p(np.abs(reward))
                # Wrapper tự động quản lý history, padding và sliding window
                action = int(wrapper.predict(obs.flatten(), float(transformed_rew)))
            else:
                # Baseline cũ (CQL, BCQ, v.v...)
                obs_contiguous = np.ascontiguousarray(obs.reshape(1, -1), dtype=np.float32)
                res = model.predict(obs_contiguous)
                action = int(res[0] if isinstance(res, (list, np.ndarray)) else res)
                
            n_t = config.action_bins[action] * min(env.queue_size, config.execution_capacity)
            obs_next, step_reward, terminated, truncated, info = env.step(action)
            
            obs = obs_next
            reward = step_reward # Cập nhật reward cho bước predict tiếp theo của DT
            ep_cost += info.get("cost", 0.0)
            tot_exec_txs += n_t
            
            if terminated or truncated:
                ep_miss = info.get("deadline_miss", False)
                if ep_miss and env.queue_size > 0:
                    # BẮT BÀI ĐÁNH TRÁO KHÁI NIỆM!
                    # Hàng chưa bán bị thanh lý ở mức phạt 500 Gwei/đơn (config.deadline_penalty)
                    penalty_cost = env.queue_size * config.deadline_penalty
                    # ep_cost += penalty_cost (đã tách riêng ở all_penalties)
                tot_miss_txs += env.queue_size
                done = True
            
        all_pure_gas.append(ep_cost)
        all_penalties.append(penalty_cost)
        all_costs.append(ep_cost + penalty_cost)
        all_misses.append(1 if ep_miss else 0)
        
        # Báo cáo tiến độ liên tục ra log (Mỗi 30 episodes báo 1 lần)
        if (i + 1) % 30 == 0 or (i + 1) == len(ep_list):
            name_str = model_name if model_name else ("Oracle" if is_oracle else "Baseline")
            print(f"  ⏳ [{name_str}] Tiến độ: {i+1}/{len(ep_list)} episodes...", flush=True)
            
    return np.mean(all_costs), np.mean(all_misses) * 100, np.mean(all_pure_gas), np.mean(all_penalties), tot_exec_txs, tot_miss_txs

def _worker_eval_model(m_path, ep_list, config, mins, maxs, target_return=475.0):
    """Worker CHỈ load model và chạy, data đã được nạp sẵn"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        model = d3rlpy.load_learnable(m_path, device=device)
        model_name = Path(m_path).name
        cost, miss, pure_gas, penalty, exec_txs, miss_txs = evaluate_policy(ep_list, model, config, mins, maxs, model_name=model_name, target_return=target_return)
        return {"name": Path(m_path).name, "cost": cost, "miss": miss, "pure_gas": pure_gas, "penalty": penalty, "exec_txs": exec_txs, "miss_txs": miss_txs, "error": None}
    except Exception as e:
        return {"name": Path(m_path).name, "cost": 0, "miss": 0, "pure_gas": 0, "penalty": 0, "exec_txs": 0, "miss_txs": 0, "error": str(e)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--data", type=str, default="Final_Project/Data/transitions_discrete_v28.parquet")
    parser.add_argument("--norm", type=str, default="Final_Project/Data/state_norm_params.json")
    parser.add_argument("--episodes", type=int, default=10) # Mặc định 10 cho nhanh
    parser.add_argument("--target", type=float, default=475.0)
    args = parser.parse_args()

    # 1. NẠP DỮ LIỆU & CẤU HÌNH (MỘT LẦN DUY NHẤT)
    print(f"[*] Nạp dữ liệu Test V28 vào RAM (Chỉ nạp 1 lần)...")
    config = TransitionBuildConfig()
    with open(args.norm, "r") as f:
        norm_data = json.load(f)
        mins, maxs = np.array(norm_data["mins"]), np.array(norm_data["maxs"])

    df = pd.read_parquet(args.data)
    
    # [ROBUSTNESS] Kiểm tra và khôi phục vật lý nếu data đang bị chuẩn hóa [0, 1]
    for i, col in enumerate(["s_queue", "s_base_fee"]): # Kiểm tra đại diện 2 cột
        if col in df.columns and df[col].max() <= 1.01:
            print(f"[!] Phát hiện dữ liệu đang ở dạng chuẩn hóa. Tiến hành khôi phục RAW...")
            from utils.offline_rl.schema import STATE_COLS
            for j, c in enumerate(STATE_COLS):
                if c in df.columns:
                    df[c] = df[c] * (maxs[j] - mins[j]) + mins[j]
            break

    unique_eps = sorted(df['episode_id'].unique())
    # ✅ CHÍNH XÁC: Lấy 20% Episodes cuối cùng (Theo thời gian thực tế)
    test_ids = unique_eps[int(len(unique_eps)*0.8):]
    selected_ids = test_ids[:args.episodes]
    
    print(f"[*] Đã trích xuất {len(test_ids)} episodes cho tập Test (20% cuối).")
    print(f"[*] Sẽ đánh giá trên {len(selected_ids)} episodes. Danh sách Episode ID:")
    print(f"    {selected_ids}")
    
    # Chỉ bốc đúng số lượng episode cần test để tiết kiệm RAM
    ep_list = [d.reset_index(drop=True) for _, d in list(df[df['episode_id'].isin(selected_ids)].groupby('episode_id'))]
    del df # Xóa dataframe gốc để giải phóng RAM

    results = []

    # 2. CHẠY BASELINES
    print("[+] Đang chạy Baselines (Greedy, Random & Oracle)...")
    
    # Dummy model class cho các Baseline cũ
    class DummyPredictor:
        def __init__(self, func):
            self.func = func
        def predict(self, obs):
            return self.func(obs)

    c_g, m_g, g_g, p_g, e_g, mt_g = evaluate_policy(ep_list, DummyPredictor(lambda x: [config.n_action_bins - 1]), config, mins, maxs)
    gas_tx_g = (g_g * len(ep_list)) / e_g if e_g > 0 else 0
    print(f"✅ XONG BASELINE: Greedy (Bán Ngay) | Chi phí: {c_g:,.0f} | Gas: {g_g:,.0f} | Phạt: {p_g:,.0f} | Trễ: {m_g:.1f}%")
    results.append(["Greedy (Bán Ngay)", f"{c_g:,.0f}", f"{g_g:,.0f}", f"{p_g:,.0f}", f"{m_g:.1f}%", "-", f"{e_g:,.0f}", f"{gas_tx_g:,.2f}", "0.0%"])

    c_r, m_r, g_r, p_r, e_r, mt_r = evaluate_policy(ep_list, DummyPredictor(lambda x: [np.random.randint(0, config.n_action_bins)]), config, mins, maxs)
    gas_tx_r = (g_r * len(ep_list)) / e_r if e_r > 0 else 0
    savings_r = ((gas_tx_g - gas_tx_r) / gas_tx_g) * 100 if gas_tx_g > 0 else 0
    print(f"✅ XONG BASELINE: Random | Chi phí: {c_r:,.0f} | Gas: {g_r:,.0f} | Phạt: {p_r:,.0f} | Trễ: {m_r:.1f}%")
    results.append(["Random (Baseline)", f"{c_r:,.0f}", f"{g_r:,.0f}", f"{p_r:,.0f}", f"{m_r:.1f}%", "-", f"{e_r:,.0f}", f"{gas_tx_r:,.2f}", f"{savings_r:,.1f}%"])

    c_o, m_o, g_o, p_o, e_o, mt_o = evaluate_policy(ep_list, None, config, mins, maxs, is_oracle=True)
    gas_tx_o = (g_o * len(ep_list)) / e_o if e_o > 0 else 0
    savings_o = ((gas_tx_g - gas_tx_o) / gas_tx_g) * 100 if gas_tx_g > 0 else 0
    print(f"✅ XONG BASELINE: Oracle (Expert) | Chi phí: {c_o:,.0f} | Gas: {g_o:,.0f} | Phạt: {p_o:,.0f} | Trễ: {m_o:.1f}%")
    results.append(["Oracle (Expert 100%)", f"{c_o:,.0f}", f"{g_o:,.0f}", f"{p_o:,.0f}", f"{m_o:.1f}%", "100.0%", f"{e_o:,.0f}", f"{gas_tx_o:,.2f}", f"{savings_o:,.1f}%"])
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
        # Khóa cứng tối đa 2 luồng để tránh treo máy (Out of Memory/CPU)
        max_workers = 2
        
        model_dir = Path(all_model_paths[0]).parent if all_model_paths else "Unknown"
        print(f"\n[+] Đang đánh giá các model từ thư mục: {model_dir}")
        print(f"[+] Tìm thấy {len(all_model_paths)} mô hình. Đang triển khai trên {max_workers} nhân CPU...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Gửi ep_list trực tiếp cho các worker
            futures = [executor.submit(_worker_eval_model, p, ep_list, config, mins, maxs, args.target) for p in all_model_paths]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res["error"]:
                    print(f"❌ Lỗi {res['name']}: {res['error']}")
                else:
                    eff = (expert_cost / res["cost"] * 100) if res["cost"] > 0 else 0
                    gas_tx = (res["pure_gas"] * len(ep_list)) / res["exec_txs"] if res["exec_txs"] > 0 else 0
                    savings_m = ((gas_tx_g - gas_tx) / gas_tx_g) * 100 if gas_tx_g > 0 else 0
                    print(f"✅ Hoàn thành: {res['name']} | Chi phí: {res['cost']:,.0f} | Gas: {res['pure_gas']:,.0f} | Phạt: {res['penalty']:,.0f} | Trễ: {res['miss']:.1f}%")
                    results.append([res["name"], f"{res['cost']:,.0f}", f"{res['pure_gas']:,.0f}", f"{res['penalty']:,.0f}", f"{res['miss']:.1f}%", f"{eff:.1f}%", f"{res['exec_txs']:,.0f}", f"{gas_tx:,.2f}", f"{savings_m:,.1f}%"])

    # 4. IN KẾT QUẢ
    headers = ["Thí Sinh", "Tổng Chi phí ↓", "Tiền Gas Thật ↓", "Tiền Phạt Ảo ↓", "Trễ (Miss) ↓", "Hiệu năng vs Expert ↑", "Số TX Xử Lý", "Gas/TX ↓", "Tiết kiệm vs Greedy ↑"]
    header_rows = results[:3]
    model_rows = sorted(results[3:], key=lambda x: float(x[5].strip('%')), reverse=True)
    
    print("\n" + "="*80)
    model_dir_str = Path(all_model_paths[0]).parent if all_model_paths else "Baselines Only"
    print(f"🏆 BẢNG VÀNG CÔNG NGHỆ Ethereum RL - PHIÊN BẢN V28 (ULTRA SPEED)")
    print(f"📁 Thư mục đang đánh giá: {model_dir_str}")
    print("="*80)
    print(tabulate(header_rows + model_rows, headers=headers, tablefmt="github"))
    print("="*80)

if __name__ == "__main__":
    main()
