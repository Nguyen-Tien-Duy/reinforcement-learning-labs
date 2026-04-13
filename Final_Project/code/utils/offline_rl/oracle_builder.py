import numpy as np
import pandas as pd
from typing import Tuple
from numba import njit
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

@njit(cache=True)
def _compute_trajectory_njit(
    gas_prices: np.ndarray, 
    incoming_requests: np.ndarray, 
    Q_max: int, 
    C_base: float, 
    C_mar: float, 
    omega: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Core DP calculation compiled with Numba for lightning speed.
    """
    T = len(gas_prices)
    if T == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int32), 0.0
        
    V = np.full((T, Q_max + 1), np.inf, dtype=np.float32)
    Policy = np.zeros((T, Q_max + 1), dtype=np.int32)
    
    # Boundary condition (at T-1, we MUST clear all pending)
    Q_array = np.arange(Q_max + 1)
    V[T-1, :] = gas_prices[T-1] * (C_base * (Q_array > 0) + C_mar * Q_array)
    Policy[T-1, :] = Q_array.astype(np.int32)
    
    # Backward Induction
    for t in range(T - 2, -1, -1):
        g_t = gas_prices[t]
        w_t = incoming_requests[t]
        for Q in range(Q_max + 1):
            n_array = np.arange(0, Q + 1)
            exec_cost = g_t * (C_base * (n_array > 0) + C_mar * n_array)
            Q_next = (Q - n_array + w_t)
            
            # Manual clip for Numba compatibility
            Q_next_clipped = np.zeros(len(Q_next), dtype=np.int32)
            for i in range(len(Q_next)):
                val = Q_next[i]
                if val < 0: val = 0
                if val > Q_max: val = Q_max
                Q_next_clipped[i] = int(val)
                
            delay_cost = omega * Q_next_clipped
            total_cost = exec_cost + delay_cost + V[t+1, Q_next_clipped]
            
            best_idx = np.argmin(total_cost)
            V[t, Q] = total_cost[best_idx]
            Policy[t, Q] = n_array[best_idx]
            
    # Forward Tracing
    optimal_n = np.zeros(T, dtype=np.int32)
    optimal_a = np.zeros(T, dtype=np.float32) 
    current_Q = 0.0
    for t in range(T):
        idx_Q = int(min(current_Q, Q_max))
        n_star = Policy[t, idx_Q]
        optimal_n[t] = n_star
        optimal_a[t] = float(n_star / idx_Q) if idx_Q > 0 else 0.0
        current_Q = idx_Q - n_star + incoming_requests[t]
        
    return optimal_a, optimal_n, float(V[0, 0])

def compute_god_view_trajectory(
    gas_prices: np.ndarray, 
    incoming_requests: np.ndarray, 
    Q_max: int = 1000, 
    C_base: float = 21000.0, 
    C_mar: float = 15000.0, 
    omega: float = 0.01
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Wrapper for the Numba-optimized trajectory solver.
    """
    return _compute_trajectory_njit(
        gas_prices.astype(np.float32), 
        incoming_requests.astype(np.float32), 
        int(Q_max), 
        float(C_base), 
        float(C_mar), 
        float(omega)
    )

def _process_episode_worker(args):
    """
    Worker function for parallel processing of episodes.
    """
    ep_df, gas_scale, C_base, C_mar, omega, oracle_ratio, suboptimal_ratio, seed_val = args
    
    # Local seed for this worker/episode
    np.random.seed(seed_val)
    
    ep_df = ep_df.sort_values("step_index")
    gas_prices_gwei = ep_df["gas_t"].to_numpy(dtype=np.float32) / gas_scale
    incoming = np.ones(len(ep_df), dtype=np.float32)
    
    opt_a, _, _ = compute_god_view_trajectory(
        gas_prices=gas_prices_gwei,
        incoming_requests=incoming,
        Q_max=1000,
        C_base=C_base,
        C_mar=C_mar,
        omega=omega
    )
    
    behavior_a = ep_df["action"].to_numpy(dtype=np.float32)
    rolls = np.random.rand(len(ep_df))
    final_actions = behavior_a.copy()
    
    oracle_mask = rolls < oracle_ratio
    final_actions[oracle_mask] = opt_a[oracle_mask]
    
    sub_mask = (rolls >= oracle_ratio) & (rolls < (oracle_ratio + suboptimal_ratio))
    final_actions[sub_mask] = 1.0 - opt_a[sub_mask]
    
    ep_df["action"] = final_actions.astype(np.float32)
    return ep_df

def apply_oracle_to_episodes(
    df: pd.DataFrame, 
    config, 
    oracle_ratio: float = 0.5,
    suboptimal_ratio: float = 0.2
) -> pd.DataFrame:
    """
    Parallelized Mix-Policy Applier using Numba and Multiprocessing.
    """
    df = df.copy()
    gas_scale = getattr(config, "gas_to_gwei_scale", 1e9)
    C_base = getattr(config, "C_base", 21000.0)
    C_mar = getattr(config, "C_mar", 15000.0)
    omega = getattr(config, "urgency_beta", 0.01)
    base_seed = getattr(config, "seed", 42)
    
    # Prepare tasks
    print(f"[+] Preparing Oracle tasks for {df['episode_id'].nunique()} episodes...")
    tasks = []
    for ep_id, ep_df in df.groupby("episode_id"):
        # We pass a unique seed per episode based on base_seed + ep_id
        try:
            ep_seed = base_seed + int(ep_id)
        except:
            ep_seed = base_seed # Fallback
            
        tasks.append((
            ep_df, gas_scale, C_base, C_mar, omega, 
            oracle_ratio, suboptimal_ratio, ep_seed
        ))
    
    # Execute in parallel
    results = []
    print(f"[+] Spawning Multiprocessing workers (Parallel JIT)...")
    with ProcessPoolExecutor() as executor:
        # We use tqdm to track progress of parallel map
        results = list(tqdm(
            executor.map(_process_episode_worker, tasks), 
            total=len(tasks), 
            desc="Optimizing Oracle Dataset"
        ))
        
    return pd.concat(results).sort_values(["episode_id", "step_index"]).reset_index(drop=True)
