import numpy as np
import pandas as pd
from typing import Tuple
import os
import psutil
from numba import njit
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

@njit
def _compute_trajectory_njit(
    gas_prices: np.ndarray, 
    incoming_requests: np.ndarray, 
    Q_max: int, 
    beta: float,
    alpha: float,
    episode_hours: float,
    reward_scale: float,
    deadline_penalty: float,
    execution_capacity: float,
    action_ratios: np.ndarray,
    s_g: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Core DP calculation compiled with Numba for lightning speed.
    STRICT DISCRETE: Only allows actions from the provided action_ratios bins.
    """
    T = len(gas_prices)
    n_bins = len(action_ratios)
    if T == 0:
        return np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.int32), np.float32(0.0)

        
    # Calculate realistic upper bound for Q and derive bin size to save memory
    total_incoming = 0.0
    for w in incoming_requests:
        total_incoming += w
        
    # Q_max is strictly bounded by the total number of transactions arrival in the episode.
    Q_max_real = float(total_incoming)
    
    # We target transaction-level precision (Q_step = 1.0).
    # We only increase Q_step if Q_max exceeds a reasonable memory limit (e.g., 20,000 states).
    MEMORY_SAFE_MAX_BINS = 20000 # change for faster test build
    if Q_max_real <= MEMORY_SAFE_MAX_BINS:
        Q_bins = int(np.ceil(Q_max_real))
        Q_step = 1.0
    else:
        Q_bins = MEMORY_SAFE_MAX_BINS
        Q_step = Q_max_real / Q_bins
    
    V = np.full((T, Q_bins + 1), np.inf, dtype=np.float32)
    # Optimization: Use uint8 for policy to save 75% RAM (n_bins is only 5)
    Policy = np.zeros((T, Q_bins + 1), dtype=np.uint8)
    
    # Boundary condition (at T-1)
    for q_idx in range(Q_bins + 1):
        Q_real = float(q_idx) * Q_step
        best_cost = np.inf
        best_bin = 0
        for b in range(n_bins):
            n = int(min(np.floor(action_ratios[b] * Q_real), execution_capacity))
            if n > Q_real:
                n = int(np.floor(Q_real))
            remaining = max(0.0, float(Q_real) - float(n))
            if remaining < 1.0:
                remaining = 0.0  # Can't have fractional transactions
            
            # Spec-aligned: Efficiency cost = n * (gas_t - gas_ref) / s_g
            # At terminal, use gas_t as both current and reference
            # Spec-aligned: Efficiency cost = n * gas_t / s_g
            cost_exec = float(n) * (gas_prices[T-1] / s_g)
            
            # Triple-Sync V27: Linear penalty (matches Env/Training)
            # Gradient preserved: partial cleanup always reduces penalty.
            cost_miss = deadline_penalty * remaining
            
            # Include delay_cost at terminal step
            time_to_deadline = episode_hours - ((T-1) * (episode_hours / T))
            time_ratio = max(0.0, min(1.0, time_to_deadline / episode_hours))
            delay_cost = beta * remaining * np.exp(alpha * (1.0 - time_ratio))
            
            total = cost_exec + cost_miss + delay_cost
            
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
                n = int(min(np.floor(action_ratios[b] * Q_real), execution_capacity))
                if n > Q_real:
                    n = int(np.floor(Q_real))
                remaining = max(0, int(Q_real) - n)
                
                # Spec-aligned: Execution cost = n * gas_t / s_g
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
    optimal_bin = np.zeros(T, dtype=np.uint8)
    optimal_n = np.zeros(T, dtype=np.int32) 
    
    current_Q = float(incoming_requests[0])
    
    for t in range(T):
        q_idx = int(np.round(current_Q / Q_step))
        if q_idx > Q_bins: q_idx = Q_bins
        if q_idx < 0: q_idx = 0
        # Quantization Fix: never round small queues to 0
        if current_Q > 0.0 and q_idx == 0:
            q_idx = 1
        
        b_star = Policy[t, q_idx]
        n_star = int(min(np.floor(action_ratios[b_star] * current_Q), execution_capacity))
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
    Q_max: int, 
    beta: float,
    alpha: float,
    episode_hours: float,
    reward_scale: float,
    deadline_penalty: float,
    execution_capacity: float,
    action_ratios: np.ndarray,
    s_g: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Wrapper for the Numba-optimized DISCRETE trajectory solver.
    Returns (optimal_bin_ids, optimal_n, total_cost).
    """
    return _compute_trajectory_njit(
        gas_prices.astype(np.float32), 
        incoming_requests.astype(np.float32), 
        int(Q_max), 
        float(beta),
        float(alpha),
        float(episode_hours),
        float(reward_scale),
        float(deadline_penalty),
        float(execution_capacity),
        action_ratios.astype(np.float32),
        float(s_g)
    )

def _process_episode_worker(args):
    """
    Worker function for parallel processing of episodes.
    V28: EPISODE-LEVEL policy assignment (mỗi episode chỉ chơi 1 chiến lược).
    - episode_policy=1 (Expert): 100% Oracle DP
    - episode_policy=2 (Medium): Epsilon-Greedy (70% Oracle + 30% Random)
    - episode_policy=3 (Random): 100% Random (KHÔNG cần chạy DP → tiết kiệm RAM)
    """
    ep_df, gas_scale, beta, alpha, ep_hours, r_scale, d_penalty, exec_cap, \
    episode_policy, seed_val, action_ratios, arr_scale, s_g = args
    
    # Local seed for this worker/episode
    np.random.seed(seed_val)
    
    ep_df = ep_df.sort_values("step_index")
    n_bins = len(action_ratios)
    T = len(ep_df)
    
    if episode_policy == 3:
        # === RANDOM POLICY ===
        # 100% random actions → SKIP DP Oracle hoàn toàn (tiết kiệm ~1GB RAM)
        final_actions = np.random.randint(0, n_bins, size=T).astype(np.int64)
        policy_labels = np.full(T, 3, dtype=np.int8)
    else:
        # === EXPERT hoặc MEDIUM → Cần chạy DP Oracle ===
        gas_prices_gwei = ep_df["gas_t"].to_numpy(dtype=np.float32) / gas_scale
        incoming = np.round(ep_df["transaction_count"].to_numpy(dtype=np.float32) * arr_scale).astype(np.int64)
        
        opt_bin, _, _ = compute_god_view_trajectory(
            gas_prices=gas_prices_gwei,
            incoming_requests=incoming,
            Q_max=10000,
            beta=beta,
            alpha=alpha,
            episode_hours=ep_hours,
            reward_scale=r_scale,
            deadline_penalty=d_penalty,
            execution_capacity=exec_cap,
            action_ratios=action_ratios,
            s_g=s_g
        )
        
        if episode_policy == 1:
            # === EXPERT POLICY === 
            # 100% Oracle - toàn bộ episode dùng hành động tối ưu DP
            final_actions = opt_bin.astype(np.int64)
            policy_labels = np.full(T, 1, dtype=np.int8)
        else:
            # === MEDIUM POLICY (Epsilon-Greedy) ===
            # 70% theo Oracle, 30% random (MỌI bước trong episode)
            EPSILON = 0.3
            rolls = np.random.rand(T)
            final_actions = opt_bin.astype(np.int64).copy()
            random_mask = rolls < EPSILON
            final_actions[random_mask] = np.random.randint(0, n_bins, size=random_mask.sum())
            policy_labels = np.full(T, 2, dtype=np.int8)
    
    ep_df["action"] = final_actions
    ep_df["policy_type"] = policy_labels
    return ep_df

def apply_oracle_to_episodes(
    df: pd.DataFrame, 
    config, 
    expert_ratio: float = 0.4,
    medium_ratio: float = 0.3,
    random_ratio: float = 0.3
) -> pd.DataFrame:
    """
    V28: Episode-Level Multi-Tier Policy Applier.
    Mỗi episode được gán NGẪU NHIÊN một chính sách duy nhất:
    - Expert (policy_type=1): 100% Oracle DP
    - Medium (policy_type=2): Epsilon-Greedy (70% Oracle + 30% Random)  
    - Random (policy_type=3): 100% Random (SKIP DP → tiết kiệm RAM)
    """
    import gc
    
    df = df.copy()
    gas_scale = config.gas_to_gwei_scale
    beta = config.urgency_beta
    alpha = config.urgency_alpha
    ep_hours = float(config.episode_hours)
    r_scale = config.reward_scale
    d_penalty = config.deadline_penalty
    exec_cap = config.execution_capacity
    s_g = getattr(config, "gas_scaling_factor", 10.0)
    base_seed = getattr(config, "seed", 42)
    action_ratios = np.array(config.action_bins, dtype=np.float32)
    arr_scale = float(getattr(config, "arrival_scale", 0.5))
    
    # 1) Gán policy theo CHIẾN THUẬT: 90% Train (Đa tầng) - 10% Eval (Chuyên gia)
    # Sắp xếp episodes theo thời gian thực (dựa trên timestamp dòng đầu tiên)
    ep_first_times = df.groupby("episode_id")[config.timestamp_col].min().sort_values()
    all_episode_ids_sorted = ep_first_times.index.tolist()
    n_episodes = len(all_episode_ids_sorted)
    
    split_idx = int(n_episodes * 0.9)
    train_episode_ids = all_episode_ids_sorted[:split_idx]
    eval_episode_ids = all_episode_ids_sorted[split_idx:]
    
    rng = np.random.RandomState(base_seed)
    episode_policies = {}
    n_expert = n_medium = n_random = 0
    
    # Gán cho tập TRAIN (Đa tầng)
    for ep_id in train_episode_ids:
        roll = rng.rand()
        if roll < expert_ratio:
            episode_policies[ep_id] = 1  # Expert
            n_expert += 1
        elif roll < expert_ratio + medium_ratio:
            episode_policies[ep_id] = 2  # Medium
            n_medium += 1
        else:
            episode_policies[ep_id] = 3  # Random
            n_random += 1
            
    # Gán cho tập EVAL (Ép 100% Expert để chấm điểm chuẩn)
    for ep_id in eval_episode_ids:
        episode_policies[ep_id] = 1
        # n_expert += 1 (Incremented below in the combined print)

    print(f"[+] V28 Chronological Policy Assignment ({n_episodes} episodes):")
    print(f"    - Train (Mixed): Expert:{n_expert}, Medium:{n_medium}, Random:{n_random}")
    print(f"    - Eval (Pure): Expert:{len(eval_episode_ids)}")
    print(f"    (Total Expert: {n_expert + len(eval_episode_ids)})")
    
    # 2) Prepare tasks with episode-level policy
    tasks = []
    for ep_id, ep_df in df.groupby("episode_id"):
        try:
            ep_seed = base_seed + int(ep_id)
        except:
            ep_seed = base_seed
        
        policy = episode_policies[ep_id]
        tasks.append((
            ep_df, gas_scale, beta, alpha, ep_hours,
            r_scale, d_penalty, exec_cap,
            policy, ep_seed, action_ratios, arr_scale, s_g
        ))
    
    # 3) Execute in parallel with RAM-aware worker limit + chunk processing
    mem_avail_gb = psutil.virtual_memory().available / (1024**3)
    max_workers_ram = int((mem_avail_gb * 0.7) / 1.0)  # 70% RAM, 1GB per worker
    max_workers_cpu = os.cpu_count() or 1
    final_workers = max(1, min(max_workers_cpu, max_workers_ram))
    
    CHUNK_SIZE = 100  # [ANTI-OOM] Xử lý từng chunk 100 episodes
    results = []
    print(f"[+] Multi-process Oracle: {final_workers} workers, chunk_size={CHUNK_SIZE}...")
    
    for chunk_start in range(0, len(tasks), CHUNK_SIZE):
        chunk_tasks = tasks[chunk_start:chunk_start + CHUNK_SIZE]
        chunk_label = f"Chunk {chunk_start//CHUNK_SIZE + 1}/{(len(tasks)-1)//CHUNK_SIZE + 1}"
        
        with ProcessPoolExecutor(max_workers=final_workers) as executor:
            chunk_results = list(tqdm(
                executor.map(_process_episode_worker, chunk_tasks), 
                total=len(chunk_tasks), 
                desc=f"[{chunk_label}] Building V28 Dataset"
            ))
        results.extend(chunk_results)
        
        # [ANTI-OOM] Dọn rác giữa mỗi chunk
        gc.collect()
        
    return pd.concat(results).sort_values(["episode_id", "step_index"]).reset_index(drop=True)
