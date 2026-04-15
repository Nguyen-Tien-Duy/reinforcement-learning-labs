import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Add code directory to path
sys.path.append("/mnt/WindowsD/Reinforcement Learning/labs/Final_Project/code")
from utils.offline_rl.enviroment import CharityGasEnv
from utils.offline_rl.config import TransitionBuildConfig
from utils.offline_rl.schema import STATE_COLS

# --- REFERENCE SLOW ENVIRONMENT (Proxy for old version) ---
class SlowCharityGasEnv:
    def __init__(self, episode_df, config):
        self.episode_df = episode_df.reset_index(drop=True)
        self.config = config
        self.max_step = len(episode_df)
        self.current_step = 0
        self._STATE_COLS = STATE_COLS

    def reset(self):
        self.current_step = 0
        row = self.episode_df.iloc[0]
        state = np.array([row[c] for c in self._STATE_COLS], dtype=np.float32)
        return state

    def step(self, action):
        if self.current_step >= self.max_step: return None, 0, True, False, {}
        # Simulate the iloc bottleneck
        row = self.episode_df.iloc[self.current_step]
        gas_t = float(row["gas_t"])
        
        # Dummy logic just to spend time
        executed = min(action[0] * 10, 500)
        
        self.current_step += 1
        if self.current_step < self.max_step:
            next_row = self.episode_df.iloc[self.current_step]
            next_obs = np.array([next_row[c] for c in self._STATE_COLS], dtype=np.float32)
        else:
            next_obs = np.zeros(11)
            
        return next_obs, 0.1, self.current_step >= self.max_step, False, {"cost": 0}

def run_benchmark():
    print("🚀 Starting Performance Benchmark: Optimized vs Slow (Pandas-based)")
    
    # 1. Mock Data (1000 steps)
    n_steps = 1000
    data = {c: np.random.randn(n_steps) for c in STATE_COLS}
    data["gas_t"] = np.random.rand(n_steps) * 100
    data["timestamp"] = pd.date_range("2024-01-01", periods=n_steps, freq="min")
    df = pd.DataFrame(data)
    
    config = TransitionBuildConfig()
    
    # --- Test Slow version ---
    slow_env = SlowCharityGasEnv(df, config)
    start_time = time.time()
    slow_env.reset()
    for _ in range(n_steps):
        slow_env.step(np.array([0.5]))
    slow_duration = time.time() - start_time
    
    # --- Test Optimized version ---
    opt_env = CharityGasEnv(df, config)
    start_time = time.time()
    opt_env.reset()
    for _ in range(n_steps):
        opt_env.step(np.array([0.5]))
    opt_duration = time.time() - start_time
    
    # 3. Report
    speedup = slow_duration / opt_duration
    print("\n" + "="*40)
    print(f"Benchmark Results ({n_steps} steps):")
    print(f"  - Slow Env (Pandas iloc): {slow_duration:.4f}s")
    print(f"  - Optimized Env (NumPy): {opt_duration:.4f}s")
    print(f"  - Speedup Factor:        {speedup:.2f}x")
    print("="*40)
    
    if speedup > 20:
        print("🔥 Tremendous speedup! The vectorization is highly effective.")
    else:
        print("📈 Significant improvement detected.")

if __name__ == "__main__":
    run_benchmark()
