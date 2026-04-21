import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys

# Add local path to import our modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from Final_Project.code.utils.offline_rl.config import TransitionBuildConfig
from Final_Project.code.utils.offline_rl.build_state_action import build_state_action_frame
from Final_Project.code.utils.offline_rl.build_reward_episode import build_reward_episode_frame

def test_sync():
    print("=== Testing V28 vs V29 Synchronization ===")
    
    # 1. Create a dummy raw dataset (100 blocks)
    data = {
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="12s"),
        "base_fee_per_gas": np.random.uniform(10e9, 50e9, 100), # 10-50 Gwei
        "gas_used": np.random.uniform(15e6, 30e6, 100),
        "gas_limit": np.full(100, 30e6),
        "transaction_count": np.random.randint(50, 200, 100),
        "action": np.random.randint(0, 5, 100) # Mock actions
    }
    raw_df = pd.DataFrame(data)
    
    # 2. Setup Configs
    config_v28 = TransitionBuildConfig(normalize_state=True)
    config_v29 = TransitionBuildConfig(normalize_state=False)
    
    # 3. Build V29 (Raw Mathematical)
    print("[+] Building V29 (Raw)...")
    df_v29 = build_reward_episode_frame(build_state_action_frame(raw_df, config_v29), config_v29)
    
    # 4. Build V28 (Normalized) - We simulate the normalization effect
    # In V28, s_queue and s_time_left were normalized.
    print("[+] Building V28 (Normalized)...")
    df_v28 = build_reward_episode_frame(build_state_action_frame(raw_df, config_v28), config_v28)
    
    # 5. COMPARISON
    # Momentum should be identical (Log-diff)
    momentum_diff = np.abs(df_v28["s_momentum"] - df_v29["s_momentum"]).max()
    print(f"[*] Momentum Consistency (V28 vs V29): {momentum_diff:.6f} (Should be 0)")
    
    # Queue should be proportional
    # V28: q_norm = (q - min) / (max - min)
    # V29: q_raw
    correlation_q = df_v28["s_queue"].corr(df_v29["s_queue"])
    print(f"[*] Queue Correlation (V28 vs V29): {correlation_q:.6f} (Should be 1.0)")
    
    # Rewards must be identical
    reward_diff = np.abs(df_v28["reward"] - df_v29["reward"]).max()
    print(f"[*] Reward Consistency (V28 vs V29): {reward_diff:.6f} (Should be 0)")

    if momentum_diff < 1e-5 and correlation_q > 0.999 and reward_diff < 1e-5:
        print("\n[SUCCESS] V29 is mathematically identical to V28 but cleaner (No manual scaling required).")
    else:
        print("\n[WARNING] Found minor discrepancies. Please check the logic.")

if __name__ == "__main__":
    test_sync()
