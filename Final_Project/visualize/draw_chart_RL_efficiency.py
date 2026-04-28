import pandas as pd
import json
import numpy as np
import d3rlpy
import torch
import argparse
import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Tự động thêm đường dẫn để tìm thấy module utils
sys.path.append(os.path.abspath("Final_Project/code"))

from utils.offline_rl.enviroment import CharityGasEnv
from utils.offline_rl.config import TransitionBuildConfig

def evaluate_policy_detailed(ep_list, algo, config, mode='model'):
    """
    Giả lập vật lý và trả về chi phí chi tiết từng episode.
    mode: 'model', 'oracle', hoặc 'naive'
    """
    costs_per_ep = []
    
    deadline_penalty = getattr(config, "deadline_penalty", 20000)
    
    for ep_df in ep_list:
        env = CharityGasEnv(ep_df, config, mins=None, maxs=None)
        
        if mode == 'oracle':
            env.expert_actions = ep_df["action"].to_numpy().astype(int)
            
        obs, _ = env.reset()
        done = False
        ep_cost = 0.0
        
        while not done:
            if mode == 'oracle':
                action = env.expert_actions[min(env.current_step, len(env.expert_actions)-1)]
            elif mode == 'naive':
                action = 4 # Execute 100%
            else:
                with torch.no_grad():
                    res = algo.predict(obs.reshape(1, -1))
                action = res.item() if hasattr(res, 'item') else res
                
            obs, reward, terminated, truncated, info = env.step(action)
            ep_cost += info.get("cost", 0.0)
            done = terminated or truncated
            
        # Cộng penalty nếu trễ hạn
        if info.get("deadline_miss", False):
            ep_cost += env.queue_size * deadline_penalty
            
        costs_per_ep.append(ep_cost)
        
    return np.array(costs_per_ep)

def draw_scientific_chart(episodes, oracle_costs, ai_costs, naive_costs, output_path):
    """Vẽ biểu đồ so sánh hiệu năng duy nhất và chuyên nghiệp"""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(episodes))
    
    # Vẽ đường Naive Baseline (Làm mốc)
    plt.plot(x, naive_costs, color='#95a5a6', linestyle='--', alpha=0.5, label='Naive (Execute Now)', linewidth=1.5)
    
    # Vẽ đường Oracle (Giới hạn tối ưu)
    plt.plot(x, oracle_costs, color='#27ae60', linestyle='-', label='Oracle (Expert DP)', linewidth=2.5, marker='o', markersize=4)
    
    # Vẽ đường AI Agent
    plt.plot(x, ai_costs, color='#2980b9', linestyle='-', label='AI Agent (DiscreteCQL)', linewidth=2.5, marker='s', markersize=4)
    
    # Đổ bóng vùng chênh lệch (Optimization Gap)
    plt.fill_between(x, oracle_costs, ai_costs, color='#3498db', alpha=0.2, label='Optimization Gap')
    
    # Tính toán thông số
    total_saved = np.sum(naive_costs) - np.sum(ai_costs)
    avg_efficiency = (np.mean(oracle_costs) / np.mean(ai_costs)) * 100
    
    # Thêm tiêu đề và nhãn
    plt.title(f'Ethereum Gas Optimization Performance (V29)\nOverall Efficiency: {avg_efficiency:.1f}% vs Oracle', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Episode Index (Testing Set)', fontsize=14)
    plt.ylabel('Total Gas Cost (Gwei)', fontsize=14)
    
    # Thêm Annotation thông tin quan trọng
    plt.text(0.02, 0.95, f"Total Saved vs Naive: {total_saved:,.0f} Gwei\nAvg Efficiency vs Oracle: {avg_efficiency:.1f}%", 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Tối ưu hóa layout
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✅ Biểu đồ đã được lưu tại: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, default="Final_Project/Data/transitions_discrete_v29.parquet")
    parser.add_argument("--episodes", type=int, default=73)
    parser.add_argument("--output", type=str, default="Final_Project/result/RL_efficiency_comparison.png")
    args = parser.parse_args()

    # 1. SETUP
    print(f"[*] Đang nạp dữ liệu V29...")
    import dataclasses
    config = dataclasses.replace(TransitionBuildConfig(), normalize_state=False)

    df = pd.read_parquet(args.data)
    
    # Logic kiểm tra RAW (đã sửa từ turn trước)
    try:
        data_dir = Path(args.data).parent
        with open(data_dir / "state_norm_params.json", "r") as f:
            params = json.load(f)
        mins_phys, max_maxs = np.array(params["mins"]), np.array(params["maxs"])
        from utils.offline_rl.schema import STATE_COLS
        for i, col in enumerate(STATE_COLS):
            if col in df.columns and df[col].max() <= 1.01:
                df[col] = df[col] * (max_maxs[i] - mins_phys[i]) + mins_phys[i]
    except Exception as e:
        print(f"⚠️ Cảnh báo: Phục hồi vật lý thất bại ({e})")

    unique_eps = sorted(df['episode_id'].unique())
    test_ids = unique_eps[int(len(unique_eps)*0.9):]
    ep_list = [d.reset_index(drop=True) for _, d in list(df[df['episode_id'].isin(test_ids[:args.episodes])].groupby('episode_id'))]
    del df

    # 2. EVALUATE
    print(f"[*] Đang giả lập Oracle...")
    oracle_costs = evaluate_policy_detailed(ep_list, None, config, mode='oracle')
    
    print(f"[*] Đang giả lập Naive Baseline (Execute Now)...")
    naive_costs = evaluate_policy_detailed(ep_list, None, config, mode='naive')
    
    print(f"[*] Đang giả lập AI Agent: {Path(args.model).name}...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    algo = d3rlpy.load_learnable(args.model, device=device)
    ai_costs = evaluate_policy_detailed(ep_list, algo, config, mode='model')

    # 3. DRAW CHART
    draw_scientific_chart(test_ids[:args.episodes], oracle_costs, ai_costs, naive_costs, args.output)

    # 4. SUMMARY TABLE
    results = [
        ["Naive (Execute Now)", f"{np.mean(naive_costs):,.0f}", "100.0% (Ref)"],
        ["AI Agent", f"{np.mean(ai_costs):,.0f}", f"{(np.mean(ai_costs)/np.mean(naive_costs)*100):.1f}%"],
        ["Oracle (Expert)", f"{np.mean(oracle_costs):,.0f}", f"{(np.mean(oracle_costs)/np.mean(naive_costs)*100):.1f}%"]
    ]
    print("\n" + "="*60)
    print("📊 TỔNG KẾT HIỆU NĂNG")
    print("="*60)
    print(tabulate(results, headers=["Policy", "Mean Cost (Gwei)", "% of Naive Cost"], tablefmt="github"))
    print("="*60)

if __name__ == "__main__":
    main()
