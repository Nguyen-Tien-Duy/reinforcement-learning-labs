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
    beta: float,
    alpha: float,
    episode_hours: float,
    reward_scale: float,
    deadline_penalty: float,
    execution_capacity: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Core DP calculation compiled with Numba for lightning speed.
    """
    T = len(gas_prices)
    if T == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int32), 0.0
        
    V = np.full((T, Q_max + 1), np.inf, dtype=np.float32)
    Policy = np.zeros((T, Q_max + 1), dtype=np.int32)
    
    # Boundary condition (at T-1, we choose between execution and penalty)
    Q_array = np.arange(Q_max + 1)
    
    for Q in range(Q_max + 1):
        # Resulting queue if we execute n: Q_final = Q - n
        # But at T-1, we only have one action. 
        # We can either execute everything or wait.
        v_q_pos = 1.0 if Q > 0 else 0.0
        cost_exec = (gas_prices[T-1] / reward_scale) * (C_base * v_q_pos + C_mar * Q)
        cost_miss = (deadline_penalty / reward_scale) * v_q_pos
        
        if cost_exec < cost_miss:
            V[T-1, Q] = cost_exec
            Policy[T-1, Q] = Q
        else:
            V[T-1, Q] = cost_miss
            Policy[T-1, Q] = 0
    
    # Backward Induction
    for t in range(T - 2, -1, -1):
        g_t = gas_prices[t]
        # In 'Arrival-before-action' logic, w_next arrives at start of next step
        w_next = incoming_requests[t+1] 
        for Q in range(Q_max + 1):
            # n_array is limited by both queue and block capacity
            n_array = np.arange(0, min(Q, int(execution_capacity)) + 1)
            # Scaled execution cost
            exec_cost = (g_t / reward_scale) * (C_base * (n_array > 0) + C_mar * n_array)
            
            # Q_next is the queue seen at start of next step (after next arrival)
            Q_next = (Q - n_array + w_next)
            
            # Manual clip for Numba compatibility
            Q_next_clipped = np.zeros(len(Q_next), dtype=np.int32)
            for i in range(len(Q_next)):
                val = Q_next[i]
                if val < 0: val = 0
                if val > Q_max: val = Q_max
                Q_next_clipped[i] = int(val)

            # Exponential urgency penalty mirroring environment logic (scaled penalty per block)
            remaining_q = Q - n_array
            time_to_deadline = episode_hours - (t * (episode_hours / T)) 
            time_ratio = max(0.0, min(1.0, time_to_deadline / episode_hours))
            delay_cost = (beta / reward_scale) * remaining_q * np.exp(alpha * (1.0 - time_ratio))
            
            total_cost = exec_cost + delay_cost + V[t+1, Q_next_clipped]
            
            best_idx = np.argmin(total_cost)
            V[t, Q] = total_cost[best_idx]
            Policy[t, Q] = n_array[best_idx]
            
    # Forward Tracing
    optimal_n = np.zeros(T, dtype=np.int32)
    optimal_a = np.zeros(T, dtype=np.float32) 
    
    # Initial state: step 0 starts with incoming_requests[0]
    current_Q = incoming_requests[0]
    
    for t in range(T):
        idx_Q = int(min(current_Q, Q_max))
        n_star = Policy[t, idx_Q]
        optimal_n[t] = n_star
        optimal_a[t] = float(n_star / idx_Q) if idx_Q > 0 else 0.0
        
        # Next step's queue: remaining plus next arrival
        if t < T - 1:
            current_Q = (current_Q - n_star) + incoming_requests[t+1]
        else:
            current_Q = current_Q - n_star
            
    return optimal_a, optimal_n, float(V[0, int(min(incoming_requests[0], Q_max))])

def compute_god_view_trajectory(
    gas_prices: np.ndarray, 
    incoming_requests: np.ndarray, 
    Q_max: int = 1000, 
    C_base: float = 21000.0, 
    C_mar: float = 15000.0, 
    beta: float = 0.01,
    alpha: float = 3.0,
    episode_hours: float = 24.0,
    reward_scale: float = 1e6,
    deadline_penalty: float = 1000.0,
    execution_capacity: float = 500.0
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
        float(beta),
        float(alpha),
        float(episode_hours),
        float(reward_scale),
        float(deadline_penalty),
        float(execution_capacity)
    )

def _process_episode_worker(args):
    """
    Worker function for parallel processing of episodes.
    """
    ep_df, gas_scale, C_base, C_mar, beta, alpha, ep_hours, r_scale, d_penalty, exec_cap, \
    oracle_ratio, suboptimal_ratio, seed_val = args
    
    # Local seed for this worker/episode
    np.random.seed(seed_val)
    
    ep_df = ep_df.sort_values("step_index")
    gas_prices_gwei = ep_df["gas_t"].to_numpy(dtype=np.float32) / gas_scale
    # Use real-world transaction arrivals
    incoming = ep_df["transaction_count"].to_numpy(dtype=np.float32)
    
    opt_a, _, _ = compute_god_view_trajectory(
        gas_prices=gas_prices_gwei,
        incoming_requests=incoming,
        beta=beta,
        alpha=alpha,
        episode_hours=ep_hours,
        reward_scale=r_scale,
        deadline_penalty=d_penalty,
        execution_capacity=exec_cap
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
    gas_scale = config.gas_to_gwei_scale
    C_base = config.C_base
    C_mar = config.C_mar
    beta = config.urgency_beta
    alpha = config.urgency_alpha
    ep_hours = float(config.episode_hours)
    r_scale = config.reward_scale
    d_penalty = config.deadline_penalty
    exec_cap = config.execution_capacity
    base_seed = getattr(config, "seed", 42) # Seed can have a default
    
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
            ep_df, gas_scale, C_base, C_mar, beta, alpha, ep_hours,
            r_scale, d_penalty, exec_cap,
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
