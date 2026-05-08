import pandas as pd
import json
import numpy as np
import d3rlpy
import torch
import re
import argparse
from pathlib import Path
from utils.offline_rl.enviroment import CharityGasEnv
from utils.offline_rl.config import TransitionBuildConfig

def analyze_model_behavior(model, ep_list, config):
    """Phân tích hành vi AI ở RAW MODE"""
    stats = []
    
    for ep_df in ep_list:
        # Tắt chuẩn hóa ở environment để AI tự dùng scaler nội bộ
        env = CharityGasEnv(ep_df, config, mins=None, maxs=None)
        obs, _ = env.reset()
        
        ep_actions = []
        ep_queues = []
        
        done = False
        while not done:
            # Lưu lại trạng thái hàng đợi thô
            ep_queues.append(env.queue_size)
            
            # Predict (d3rlpy tự lo chuẩn hóa nội bộ từ RAW obs)
            res = model.predict(obs.reshape(1, -1))
            action = res.item() if hasattr(res, 'item') else res
            ep_actions.append(int(action))
            
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
        stats.append({
            "actions": ep_actions,
            "queues": ep_queues
        })
    
    # Tính toán thống kê gộp trên toàn bộ test episodes
    all_a = np.concatenate([s["actions"] for s in stats])
    all_q = np.concatenate([s["queues"] for s in stats])
    
    # Correlation (Hàng đợi vs Hành động)
    if np.std(all_q) > 0 and np.std(all_a) > 0:
        corr_q_a = np.corrcoef(all_q, all_a)[0, 1]
    else:
        corr_q_a = 0.0
        
    avg_a = np.mean(all_a)
    
    # Tỉ lệ các hành động (%)
    unique, counts = np.unique(all_a, return_counts=True)
    dist = dict(zip(unique, counts / len(all_a) * 100))
    
    return avg_a, corr_q_a, dist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, required=True, help="Thư mục chứa các file .d3")
    parser.add_argument("--data", type=str, default="Final_Project/Data/transitions_discrete_v28.parquet")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    # 1. SETUP
    print(f"[*] Đang nạp dữ liệu chẩn đoán RAW MODE...")
    import dataclasses
    config_base = TransitionBuildConfig()
    config = dataclasses.replace(config_base, normalize_state=False)

    # 2. LOAD DATA
    df = pd.read_parquet(args.data)
    unique_eps = sorted(df['episode_id'].unique())
    test_ids = unique_eps[int(len(unique_eps)*0.9):]
    ep_list = [d.reset_index(drop=True) for _, d in list(df[df['episode_id'].isin(test_ids[:args.episodes])].groupby('episode_id'))]
    del df

    # 3. LIST MODELS
    p = Path(args.models_dir)
    model_paths = sorted(list(p.glob("model_*.d3")), key=lambda x: int(re.search(r'model_(\d+)', x.name).group(1)))

    results = []
    print(f"[+] Bắt đầu chẩn đoán RAW cho {len(model_paths)} checkpoints...")
    
    for m_path in model_paths:
        try:
            # Chạy CPU cho chẩn đoán để tránh tranh chấp VRAM
            model = d3rlpy.load_learnable(str(m_path), device="cpu")
            avg_a, corr_q_a, dist = analyze_model_behavior(model, ep_list, config)
            
            step = int(re.search(r'model_(\d+)', m_path.name).group(1))
            results.append({
                "step": step,
                "avg_action": avg_a,
                "corr_queue_action": corr_q_a,
                "A0": dist.get(0, 0),
                "A4": dist.get(4, 0)
            })
            print(f"✅ RAW Checked {m_path.name}: Avg Action: {avg_a:.2f} | Corr Q-A: {corr_q_a:.3f}")
        except Exception as e:
            print(f"❌ Lỗi {m_path.name}: {e}")

    # 4. REPORT
    res_df = pd.DataFrame(results)
    res_df.to_csv("mass_diagnosis_raw_report.csv", index=False)
    
    print("\n" + "="*50)
    print("🏆 BÁO CÁO CHẨN ĐOÁN CHIẾN THUẬT AI (RAW MODE)")
    print("="*50)
    print(res_df[["step", "avg_action", "corr_queue_action", "A0", "A4"]].to_string(index=False))
    print("\n[*] Đã lưu báo cáo chi tiết vào mass_diagnosis_raw_report.csv")

if __name__ == "__main__":
    main()
