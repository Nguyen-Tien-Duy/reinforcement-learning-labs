import numpy as np
import pandas as pd
from typing import Tuple
from numba import njit
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

@njit
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
    execution_capacity: float,
    action_ratios: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Core DP calculation compiled with Numba for lightning speed.
    STRICT DISCRETE: Only allows actions from the provided action_ratios bins.
    """
    T = len(gas_prices)
    n_bins = len(action_ratios)
    if T == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), 0.0
        
    # Calculate realistic upper bound for Q and derive bin size to save memory
    total_incoming = 0.0
    for w in incoming_requests:
        total_incoming += w
    Q_max_real = float(total_incoming) + 5000.0
    
    # Dynamic binning: Ensure Q_step is exactly 1.0 for high precision,
    # but cap Q_bins to avoid RAM overflow. 
    # For V20, Q_max_real is ~10k-20k, so 15k bins is plenty.
    Q_bins = int(min(15000, Q_max_real))
    Q_step = max(1.0, Q_max_real / Q_bins)
    
    V = np.full((T, Q_bins + 1), np.inf, dtype=np.float32)
    # Policy now stores the BIN INDEX (0..n_bins-1) instead of raw n
    Policy = np.zeros((T, Q_bins + 1), dtype=np.int32)
    
    # Boundary condition (at T-1)
    for q_idx in range(Q_bins + 1):
        Q_real = float(q_idx) * Q_step
        best_cost = np.inf
        best_bin = 0
        for b in range(n_bins):
            n = int(min(np.round(action_ratios[b] * Q_real), execution_capacity))
            if n > Q_real:
                n = int(np.floor(Q_real))
            remaining = max(0.0, float(Q_real) - float(n))
            if remaining < 1.0:
                remaining = 0.0  # Can't have fractional transactions
            
            # Spec-aligned: Efficiency cost = n * (gas_t - gas_ref) / s_g
            # At terminal, use gas_t as both current and reference
            s_g = 10.0
            cost_exec = float(n) * (gas_prices[T-1] / s_g)  # Cost of executing at current gas
            # Terminal penalty for remaining queue
            v_rem_pos = 1.0 if remaining > 0 else 0.0
            cost_miss = deadline_penalty * v_rem_pos
            total = cost_exec + cost_miss
            
            if total < best_cost:
                best_cost = total
                best_bin = b
                
        V[T-1, q_idx] = best_cost
        Policy[T-1, q_idx] = best_bin
    
    # Backward Induction
    for t in range(T - 2, -1, -1):
        g_t = gas_prices[t]
        w_next = float(incoming_requests[t+1])
        for q_idx in range(Q_bins + 1):
            Q_real = float(q_idx) * Q_step
            best_cost = np.inf
            best_bin = 0
            for b in range(n_bins):
                n = int(min(np.round(action_ratios[b] * Q_real), execution_capacity))
                if n > Q_real:
                    n = int(np.floor(Q_real))
                remaining = max(0.0, float(Q_real) - float(n))
                if remaining < 1.0:
                    remaining = 0.0  # Can't have fractional transactions
                
                # Spec-aligned: Execution cost = n * gas_t / s_g (no C_base!)
                s_g = 10.0
                exec_cost = float(n) * (g_t / s_g)
                
                # Next queue indexing - Use CEIL to ensure any non-zero queue 
                # triggers the penalty/cost states.
                Q_next_real = remaining + w_next
                q_next_idx = int(np.ceil(Q_next_real / Q_step))
                if q_next_idx < 0: q_next_idx = 0
                if q_next_idx > Q_bins: q_next_idx = Q_bins

                # Urgency penalty (Linear scale - NO CLIPPING)
                # Outliers handled via Robust Normalization in training.
                time_to_deadline = episode_hours - (t * (episode_hours / T)) 
                time_ratio = max(0.0, min(1.0, time_to_deadline / episode_hours))
                delay_cost = beta * remaining * np.exp(alpha * (1.0 - time_ratio))
                
                total = exec_cost + delay_cost + V[t+1, q_next_idx]
                if total < best_cost:
                    best_cost = total
                    best_bin = b
            
            V[t, q_idx] = best_cost
            Policy[t, q_idx] = best_bin
            
    # Forward Tracing — returns BIN IDs
    optimal_bin = np.zeros(T, dtype=np.int32)
    optimal_n = np.zeros(T, dtype=np.int32) 
    
    current_Q = float(incoming_requests[0])
    
    for t in range(T):
        q_idx = int(np.round(current_Q / Q_step))
        if q_idx > Q_bins: q_idx = Q_bins
        if q_idx < 0: q_idx = 0
        
        b_star = Policy[t, q_idx]
        n_star = int(min(np.round(action_ratios[b_star] * current_Q), execution_capacity))
        if n_star > current_Q:
            n_star = int(np.floor(current_Q))
        # Clear fractional remainder
        if (current_Q - float(n_star)) < 1.0:
            n_star = int(np.ceil(current_Q))
            
        optimal_bin[t] = b_star
        optimal_n[t] = n_star
        
        if t < T - 1:
            current_Q = max(0.0, current_Q - float(n_star)) + float(incoming_requests[t+1])
        else:
            current_Q = max(0.0, current_Q - float(n_star))
            
    q0_idx = int(np.round(float(incoming_requests[0]) / Q_step))
    if q0_idx > Q_bins: q0_idx = Q_bins
    if q0_idx < 0: q0_idx = 0
    return optimal_bin, optimal_n, float(V[0, q0_idx])

def compute_god_view_trajectory(
    gas_prices: np.ndarray, 
    incoming_requests: np.ndarray, 
    Q_max: int = 1000, 
    C_base: float = 21000.0, 
    C_mar: float = 15000.0, 
    beta: float = 0.01,
    alpha: float = 3.0,
    episode_hours: float = 24.0,
    reward_scale: float = 1e9,
    deadline_penalty: float = 5000000000.0,
    execution_capacity: float = 500.0,
    action_ratios: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Wrapper for the Numba-optimized DISCRETE trajectory solver.
    Returns (optimal_bin_ids, optimal_n, total_cost).
    """
    if action_ratios is None:
        action_ratios = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
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
        float(execution_capacity),
        action_ratios.astype(np.float32)
    )

def _process_episode_worker(args):
    """
    Worker function for parallel processing of episodes.
    Now operates in DISCRETE action space (bin IDs).
    """
    ep_df, gas_scale, C_base, C_mar, beta, alpha, ep_hours, r_scale, d_penalty, exec_cap, \
    oracle_ratio, suboptimal_ratio, seed_val, action_ratios, arr_scale = args
    
    # Local seed for this worker/episode
    np.random.seed(seed_val)
    
    ep_df = ep_df.sort_values("step_index")
    gas_prices_gwei = ep_df["gas_t"].to_numpy(dtype=np.float32) / gas_scale
    incoming = ep_df["transaction_count"].to_numpy(dtype=np.float32) * arr_scale
    
    n_bins = len(action_ratios)
    
    opt_bin, _, _ = compute_god_view_trajectory(
        gas_prices=gas_prices_gwei,
        incoming_requests=incoming,
        beta=beta,
        alpha=alpha,
        episode_hours=ep_hours,
        reward_scale=r_scale,
        deadline_penalty=d_penalty,
        execution_capacity=exec_cap,
        action_ratios=action_ratios
    )
    
    behavior_a = ep_df["action"].to_numpy(dtype=np.int64)
    rolls = np.random.rand(len(ep_df))
    final_actions = behavior_a.copy()
    
    # Oracle actions: use DP-computed optimal bin IDs
    oracle_mask = rolls < oracle_ratio
    final_actions[oracle_mask] = opt_bin[oracle_mask]
    
    # Suboptimal: invert the bin ID (bin 0 <-> bin 4, bin 1 <-> bin 3, etc.)
    sub_mask = (rolls >= oracle_ratio) & (rolls < (oracle_ratio + suboptimal_ratio))
    final_actions[sub_mask] = (n_bins - 1) - opt_bin[sub_mask]
    
    ep_df["action"] = final_actions.astype(np.int64)
    return ep_df

def apply_oracle_to_episodes(
    df: pd.DataFrame, 
    config, 
    oracle_ratio: float = 0.5,
    suboptimal_ratio: float = 0.2
) -> pd.DataFrame:
    """
    Parallelized Mix-Policy Applier using Numba and Multiprocessing.
    Now operates in DISCRETE action space (bin IDs).
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
    base_seed = getattr(config, "seed", 42)
    action_ratios = np.array(config.action_bins, dtype=np.float32)
    arr_scale = float(getattr(config, "arrival_scale", 0.5))
    
    # Prepare tasks
    print(f"[+] Preparing Discrete Oracle tasks for {df['episode_id'].nunique()} episodes ({len(action_ratios)} bins)...")
    print(f"    arrival_scale={arr_scale}, exec_cap={exec_cap}")
    tasks = []
    for ep_id, ep_df in df.groupby("episode_id"):
        try:
            ep_seed = base_seed + int(ep_id)
        except:
            ep_seed = base_seed
            
        tasks.append((
            ep_df, gas_scale, C_base, C_mar, beta, alpha, ep_hours,
            r_scale, d_penalty, exec_cap,
            oracle_ratio, suboptimal_ratio, ep_seed, action_ratios, arr_scale
        ))
    
    # Execute in parallel
    results = []
    print(f"[+] Spawning Multiprocessing workers (Parallel JIT)...")
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(_process_episode_worker, tasks), 
            total=len(tasks), 
            desc="Optimizing Discrete Oracle Dataset"
        ))
        
    return pd.concat(results).sort_values(["episode_id", "step_index"]).reset_index(drop=True)
