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
        if info.get("deadline_miss", False):
            ep_cost += env.queue_size * deadline_penalty
        costs_per_ep.append(ep_cost)
    return np.array(costs_per_ep)

def draw_cumulative_chart(episodes, oracle_costs, ai_costs, naive_costs, output_path):
    """Vẽ biểu đồ tích lũy để lộ rõ Optimization Gap"""
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1, 1.2]})
    
    x = np.arange(len(episodes))
    
    # --- PLOT 1: DAILY COST (Dùng Log Scale để thấy chênh lệch nhỏ) ---
    ax1.plot(x, naive_costs, color='#95a5a6', linestyle='--', alpha=0.6, label='Naive (Execute Now)')
    ax1.plot(x, oracle_costs, color='#27ae60', label='Oracle (Expert DP)', linewidth=2)
    ax1.plot(x, ai_costs, color='#2980b9', label='AI Agent (CQL)', linewidth=2)
    ax1.set_yscale('log') # Quan trọng: Log scale giúp thấy chênh lệch ở vùng giá thấp
    ax1.set_title('Daily Gas Cost Comparison (Log Scale)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Gas Cost (Gwei) - Log Scale', fontsize=12)
    ax1.legend()

    # --- PLOT 2: CUMULATIVE SAVINGS (Nơi lộ rõ GAP) ---
    cum_naive = np.cumsum(naive_costs)
    cum_oracle = np.cumsum(oracle_costs)
    cum_ai = np.cumsum(ai_costs)
    
    # Tính toán "Tiền tiết kiệm được" so với Naive
    savings_oracle = cum_naive - cum_oracle
    savings_ai = cum_naive - cum_ai
    
    ax2.fill_between(x, 0, savings_oracle, color='#2ecc71', alpha=0.3, label='Potential Savings (Oracle)')
    ax2.plot(x, savings_oracle, color='#27ae60', linewidth=3, label='Max Possible Savings (Oracle)')
    ax2.plot(x, savings_ai, color='#2980b9', linewidth=3, label='Actual AI Savings')
    
    ax2.set_title('Cumulative Gas Savings vs. Naive Baseline', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Episode Index', fontsize=12)
    ax2.set_ylabel('Total Gwei Saved', fontsize=12)
    
    # Annotate final values
    final_savings_ai = savings_ai[-1]
    final_savings_oracle = savings_oracle[-1]
    ax2.annotate(f'AI Saved: {final_savings_ai:,.0f} Gwei', xy=(x[-1], final_savings_ai), xytext=(-120, 20),
                 textcoords='offset points', arrowprops=dict(arrowstyle='->', color='#2980b9'), fontweight='bold', color='#2980b9')
    ax2.annotate(f'Oracle Max: {final_savings_oracle:,.0f} Gwei', xy=(x[-1], final_savings_oracle), xytext=(-120, 40),
                 textcoords='offset points', arrowprops=dict(arrowstyle='->', color='#27ae60'), fontweight='bold', color='#27ae60')

    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✅ Biểu đồ tích lũy đã được lưu tại: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, default="Final_Project/Data/transitions_discrete_v29.parquet")
    parser.add_argument("--episodes", type=int, default=73)
    parser.add_argument("--output", type=str, default="Final_Project/result/RL_cumulative_gap_v29.png")
    args = parser.parse_args()

    # 1. SETUP
    print(f"[*] Đang nạp dữ liệu V29...")
    import dataclasses
    config = dataclasses.replace(TransitionBuildConfig(), normalize_state=False)
    df = pd.read_parquet(args.data)
    
    # Logic kiểm tra RAW
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
    print(f"[*] Đang giả lập các chiến thuật...")
    oracle_costs = evaluate_policy_detailed(ep_list, None, config, mode='oracle')
    naive_costs = evaluate_policy_detailed(ep_list, None, config, mode='naive')
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    algo = d3rlpy.load_learnable(args.model, device=device)
    ai_costs = evaluate_policy_detailed(ep_list, algo, config, mode='model')

    # 3. DRAW CHART
    draw_cumulative_chart(test_ids[:args.episodes], oracle_costs, ai_costs, naive_costs, args.output)

if __name__ == "__main__":
    main()
