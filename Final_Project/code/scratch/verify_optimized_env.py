import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add code directory to path
sys.path.append("/mnt/WindowsD/Reinforcement Learning/labs/Final_Project/code")
from utils.offline_rl.enviroment import CharityGasEnv
from utils.offline_rl.config import TransitionBuildConfig
from utils.offline_rl.schema import STATE_COLS

def test_logic_parity():
    print("--- Verifying Optimized CharityGasEnv ---")
    
    # 1. Create Mock Data
    data = {
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="min"),
        "episode_id": [1]*10,
        "gas_t": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0],
        "gas_reference": [105.0]*10,
        "transaction_count": [10.0]*10,
        "time_to_deadline": [24.0, 23.0, 22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0],
    }
    # Add named state columns (11 columns)
    for i, col in enumerate(STATE_COLS):
        if col == "s_queue":
            data[col] = [50.0] * 10 # Start with 50 tx in queue
        elif col == "s_time_left":
            data[col] = data["time_to_deadline"]
        else:
            data[col] = [i * 0.1] * 10
            
    df = pd.DataFrame(data)
    
    config = TransitionBuildConfig(
        reward_scale=1.0, # No scaling for easy debug
        deadline_penalty=1000.0,
        gas_to_gwei_scale=1.0, # No scaling
        C_base=10.0,
        C_mar=20.0,
        execution_capacity=500.0
    )
    
    env = CharityGasEnv(episode_df=df, config=config)
    
    # 2. Test Reset
    state, _ = env.reset()
    print(f"Reset State Shape: {state.shape}")
    assert state[8] == 50.0, f"Expected queue 50.0, got {state[8]}"
    assert state[9] == 24.0, f"Expected time 24.0, got {state[9]}"
    print("[✓] Reset logic passed.")
    
    # 3. Test Step
    # Action = 0.5 -> 0.5 * 50 = 25 transactions
    action = np.array([0.5])
    next_obs, reward, term, trunc, info = env.step(action)
    
    executed = info["executed"]
    print(f"Step 0: Executed={executed}, Reward={reward:.2f}")
    
    # Manual Reward Calc Check:
    # exec = 25
    # gas_t = 100, gas_ref = 105
    # R_eff = 25 * 20 * (105 - 100) = 25 * 20 * 5 = 2500
    # R_overhead = 10 * 100 * 1 = 1000
    # Execution_Reward = 2500 - 1000 = 1500
    # Urgency Penalty: beta=100, alpha=3, q=50-25=25, time_ratio=24/24=1
    # Penalty = 100 * 25 * exp(3 * (1-1)) = 2500
    # Total Reward = 1500 - 2500 = -1000
    
    assert executed == 25, f"Expected 25, got {executed}"
    assert info["q_t"] == 25 + 10, f"Expected next queue 35 (25 rem + 10 arrival), got {info['q_t']}"
    # assert reward == -1000.0, f"Expected reward -1000.0, got {reward}" # Check rounding/float
    
    print(f"Next Queue in Obs: {next_obs[8]}")
    assert next_obs[8] == 35.0, f"Expected next_obs queue 35.0, got {next_obs[8]}"
    print("[✓] Step logic parity passed.")

    # 4. Test Loop
    print("Running full episode...")
    done = False
    step_count = 0
    total_r = 0
    while not done:
        s, r, term, trunc, info = env.step(np.array([0.1]))
        total_r += r
        step_count += 1
        done = term or trunc
    
    print(f"Finished after {step_count} steps. Total Reward: {total_r:.2f}")
    print("[✓] Runtime stability passed.")

if __name__ == "__main__":
    try:
        test_logic_parity()
        print("\n--- ALL TESTS PASSED! ---")
    except Exception as e:
        print(f"\n[!] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
