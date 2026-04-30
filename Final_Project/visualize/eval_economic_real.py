import pandas as pd
import numpy as np
import d3rlpy
import sys
import os
from tabulate import tabulate

sys.path.append(os.path.abspath("Final_Project/code"))
from utils.offline_rl.enviroment import CharityGasEnv
from utils.offline_rl.config import TransitionBuildConfig

def evaluate_economic_real(model_path, data_path):
    config = TransitionBuildConfig()
    df = pd.read_parquet(data_path)
    
    unique_eps = sorted(df['episode_id'].unique())
    test_ids = unique_eps[int(len(unique_eps)*0.8):][:100] # Chạy 100 tập cho nhanh
    
    model = d3rlpy.load_learnable(model_path)
    
    results = []
    
    # Hàm Helper
    def run_policy(name, policy_fn, is_oracle=False):
        tot_gas = 0.0
        tot_exec_txs = 0
        tot_miss_txs = 0
        
        for ep_id in test_ids:
            ep_df = df[df['episode_id'] == ep_id].reset_index(drop=True)
            env = CharityGasEnv(ep_df, config, mins=None, maxs=None)
            
            if is_oracle:
                env.expert_actions = ep_df["action"].to_numpy().astype(int)
                
            obs, _ = env.reset()
            done = False
            
            while not done:
                if is_oracle:
                    action = env.expert_actions[min(env.current_step, len(env.expert_actions)-1)]
                else:
                    obs_contig = np.ascontiguousarray(obs.reshape(1, -1), dtype=np.float32)
                    action = int(policy_fn(obs_contig)[0]) if policy_fn else np.random.randint(0, 5)
                    
                # Lấy số lượng thực sự đã xử lý
                n_t = config.action_bins[action] * min(env.queue_size, config.execution_capacity)
                obs, reward, terminated, truncated, info = env.step(action)
                
                tot_gas += info.get("cost", 0.0)
                tot_exec_txs += n_t
                
                if terminated or truncated:
                    tot_miss_txs += env.queue_size
                    done = True
                    
        avg_gas_per_tx = tot_gas / tot_exec_txs if tot_exec_txs > 0 else 0
        results.append([
            name, 
            f"{tot_gas:,.0f}", 
            f"{tot_exec_txs:,}", 
            f"{tot_miss_txs:,}", 
            f"{avg_gas_per_tx:,.2f}"
        ])

    print("\n[*] Đang chạy Baseline (Random)...")
    run_policy("Random", None, False)
    
    print("[*] Đang chạy AI Model...")
    run_policy("AI Model 500k", model.predict, False)
    
    print("[*] Đang chạy Oracle...")
    run_policy("Oracle", None, True)
    
    print("\n" + "="*80)
    print("🏆 BÓC TÁCH KINH TẾ THỰC SỰ: CHI PHÍ GAS TRÊN MỖI GIAO DỊCH (Gwei/tx)")
    print("="*80)
    headers = ["Thí Sinh", "Tổng Tiền Gas (Gwei)", "Số TX Đã Xử Lý", "Số TX Bỏ Rơi (Miss)", "Đơn giá Gas/TX (Gwei/tx) ↓"]
    print(tabulate(results, headers=headers, tablefmt="github"))
    print("="*80)

if __name__ == "__main__":
    m_path = "d3rlpy_logs/DiscreteCQL_V6_20260429_1119_20260429111912/model_500000.d3"
    d_path = "Final_Project/Data/transitions_v33_L2_Batching_RAW.parquet"
    evaluate_economic_real(m_path, d_path)
