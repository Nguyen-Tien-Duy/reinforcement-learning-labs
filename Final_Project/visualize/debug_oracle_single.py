"""
Debug Oracle DP on a single episode.
Shows exactly what the Oracle is thinking at each step.
"""
import numpy as np
import pandas as pd
import sys

def debug_oracle_episode(raw_parquet, episode_idx=0):
    print(f"[+] Loading raw data: {raw_parquet}")
    raw = pd.read_parquet(raw_parquet)
    
    # Build episodes the same way as the pipeline
    raw['timestamp'] = pd.to_datetime(raw['timestamp'], utc=True)
    raw = raw.sort_values('timestamp')
    min_ts = raw['timestamp'].min()
    elapsed = (raw['timestamp'] - min_ts).dt.total_seconds().astype(np.int64)
    raw['episode_id'] = elapsed // (24 * 3600)
    
    episodes = sorted(raw['episode_id'].unique())
    ep_id = episodes[episode_idx]
    ep = raw[raw['episode_id'] == ep_id].copy()
    
    print(f"[+] Episode {ep_id}: {len(ep)} steps")
    
    # Oracle parameters (must match training config!)
    gas_scale = 1e9
    arrival_scale = 0.05
    exec_cap = 500.0
    beta = 50.0
    alpha = 3.0
    episode_hours = 24.0
    deadline_penalty = 1_000_000.0
    action_ratios = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    s_g = 10.0
    
    gas_prices = (ep['base_fee_per_gas'].values.astype(np.float64) / gas_scale).astype(np.float32)
    incoming = (ep['transaction_count'].values.astype(np.float64) * arrival_scale).astype(np.float32)
    
    T = len(gas_prices)
    n_bins = len(action_ratios)
    
    # Setup DP grid
    total_incoming = float(incoming.sum())
    Q_max_real = total_incoming + 5000.0
    Q_bins = int(min(15000, Q_max_real))
    Q_step = max(1.0, Q_max_real / Q_bins)
    
    print(f"[+] DP Grid: Q_max_real={Q_max_real:.1f}, Q_bins={Q_bins}, Q_step={Q_step:.4f}")
    print(f"[+] Total incoming TX: {total_incoming:.1f}")
    print(f"[+] Deadline penalty: {deadline_penalty}")
    print(f"[+] Exec capacity: {exec_cap}")
    
    V = np.full((T, Q_bins + 1), np.inf, dtype=np.float64)
    Policy = np.zeros((T, Q_bins + 1), dtype=np.int32)
    
    # ========== BOUNDARY CONDITION (T-1) ==========
    print(f"\n{'='*70}")
    print(f"BOUNDARY CONDITION (step T-1 = {T-1})")
    print(f"{'='*70}")
    
    for q_idx in range(Q_bins + 1):
        Q_real = float(q_idx) * Q_step
        best_cost = np.inf
        best_bin = 0
        for b in range(n_bins):
            n = int(min(np.round(action_ratios[b] * Q_real), exec_cap))
            if n > Q_real:
                n = int(np.floor(Q_real))
            remaining = max(0.0, Q_real - float(n))
            if remaining < 1.0:
                remaining = 0.0
            
            cost_exec = float(n) * (gas_prices[T-1] / s_g)
            v_rem_pos = 1.0 if remaining > 0 else 0.0
            cost_miss = deadline_penalty * v_rem_pos
            total = cost_exec + cost_miss
            
            if total < best_cost:
                best_cost = total
                best_bin = b
        
        V[T-1, q_idx] = best_cost
        Policy[T-1, q_idx] = best_bin
        
        # Log a few representative q_idx values
        if q_idx in [0, 1, 2, 5, 10, 50, 100]:
            print(f"  q_idx={q_idx:>4} (Q={Q_real:>8.1f}): Policy=Action {best_bin}, V={best_cost:.4f}")
    
    # ========== BACKWARD INDUCTION ==========
    for t in range(T - 2, -1, -1):
        g_t = gas_prices[t]
        w_next = float(incoming[t+1])
        for q_idx in range(Q_bins + 1):
            Q_real = float(q_idx) * Q_step
            best_cost = np.inf
            best_bin = 0
            for b in range(n_bins):
                n = int(min(np.round(action_ratios[b] * Q_real), exec_cap))
                if n > Q_real:
                    n = int(np.floor(Q_real))
                remaining = max(0.0, Q_real - float(n))
                if remaining < 1.0:
                    remaining = 0.0
                
                exec_cost = float(n) * (g_t / s_g)
                
                Q_next_real = remaining + w_next
                q_next_idx = int(np.ceil(Q_next_real / Q_step))
                if q_next_idx < 0: q_next_idx = 0
                if q_next_idx > Q_bins: q_next_idx = Q_bins
                
                time_to_deadline = episode_hours - (t * (episode_hours / T))
                time_ratio = max(0.0, min(1.0, time_to_deadline / episode_hours))
                delay_cost = beta * remaining * np.exp(alpha * (1.0 - time_ratio))
                
                total = exec_cost + delay_cost + V[t+1, q_next_idx]
                if total < best_cost:
                    best_cost = total
                    best_bin = b
            
            V[t, q_idx] = best_cost
            Policy[t, q_idx] = best_bin
    
    # ========== FORWARD TRACE ==========
    print(f"\n{'='*70}")
    print(f"FORWARD TRACE (actual trajectory)")
    print(f"{'='*70}")
    print(f"{'Step':>5} {'Queue':>8} {'q_idx':>6} {'Policy':>7} {'Exec':>6} {'Remain':>8} {'Gas':>10}")
    print(f"{'-'*60}")
    
    current_Q = float(incoming[0])
    miss_at_end = False
    
    for t in range(T):
        q_idx = int(np.round(current_Q / Q_step))
        if q_idx > Q_bins: q_idx = Q_bins
        if q_idx < 0: q_idx = 0
        
        b_star = Policy[t, q_idx]
        n_star = int(min(np.round(action_ratios[b_star] * current_Q), exec_cap))
        if n_star > current_Q:
            n_star = int(np.floor(current_Q))
        
        remaining = max(0.0, current_Q - float(n_star))
        if remaining < 1.0:
            n_star = int(np.ceil(current_Q))
            remaining = 0.0
        
        # Print last 10 steps and first 5
        if t < 5 or t >= T - 10:
            print(f"{t:>5} {current_Q:>8.1f} {q_idx:>6} {'A'+str(b_star):>7} {n_star:>6} {remaining:>8.1f} {gas_prices[t]:>10.6f}")
        elif t == 5:
            print(f"  ... (skipping middle steps) ...")
        
        if t < T - 1:
            current_Q = max(0.0, current_Q - float(n_star)) + float(incoming[t+1])
        else:
            current_Q = max(0.0, current_Q - float(n_star))
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULT: Queue after last action = {current_Q:.1f}")
    if current_Q > 0:
        print(f"❌ MISS! Oracle left {current_Q:.1f} items in queue")
    else:
        print(f"✅ SUCCESS! Oracle cleared the queue completely")
    print(f"{'='*70}")

if __name__ == "__main__":
    raw_path = sys.argv[1] if len(sys.argv) > 1 else 'Final_Project/Data/data_2024-04-10_2026-04-10.parquet'
    ep_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    debug_oracle_episode(raw_path, ep_idx)
