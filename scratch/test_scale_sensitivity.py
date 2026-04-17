
import pandas as pd
import numpy as np
from pathlib import Path
from utils.load_data import TransitionBuildConfig, build_transitions_from_parquet

def run_sensitivity_test(scale, beta):
    config = TransitionBuildConfig(
        arrival_scale=scale, 
        urgency_beta=beta, 
        deadline_penalty=500.0,
        normalize_state=True
    )
    
    raw_path = Path("Final_Project/Data/data_2024-04-10_2026-04-10.parquet")
    df_raw = pd.read_parquet(raw_path).head(1000000)
    
    tmp_raw = Path("scratch/tmp_test_raw.parquet")
    df_raw.to_parquet(tmp_raw)
    
    df_processed = build_transitions_from_parquet(
        tmp_raw,
        config,
        use_oracle=True,
        oracle_ratio=1.0,
        suboptimal_ratio=0.0,
        config_hash=f"sensitivity_{scale}_{beta}"
    )
    
    dist = df_processed['action'].value_counts(normalize=True).sort_index()
    
    results = []
    for ep_id, ep_df in df_processed.groupby("episode_id"):
        gas_prices_gwei = ep_df["gas_t"].to_numpy() / 1e9
        q_size = ep_df["queue_size"].to_numpy()
        actions = ep_df["action"].to_numpy()
        bins = np.array(config.action_bins)
        exec_cap = config.execution_capacity
        s_g = config.gas_scaling_factor
        
        oracle_gas_cost = 0
        dumb_gas_cost = 0
        total_holding_cost = 0
        
        current_dumb_q = q_size[0]
        
        for t in range(len(ep_df)):
            g_t = gas_prices_gwei[t]
            
            # Oracle
            ratio_opt = bins[actions[t]]
            n_opt = min(np.floor(ratio_opt * q_size[t]), exec_cap)
            exec_cost = n_opt * (g_t / s_g)
            oracle_gas_cost += exec_cost
            
            # Holding cost (Simplified for metrics)
            remaining = q_size[t] - n_opt
            time_ratio = 1.0 - (t / len(ep_df))
            holding = beta * remaining * np.exp(3.0 * (1.0 - time_ratio))
            total_holding_cost += holding
            
            # Dumb
            n_dumb = min(np.floor(1.0 * current_dumb_q), exec_cap)
            dumb_gas_cost += n_dumb * (g_t / s_g)
            
            if t < len(ep_df) - 1:
                arrival = q_size[t+1] - (q_size[t] - n_opt)
                current_dumb_q = max(0, current_dumb_q - n_dumb + arrival)

        last_row = ep_df.iloc[-1]
        last_ratio = bins[actions[-1]]
        last_n = min(np.floor(last_ratio * last_row["queue_size"]), exec_cap)
        missed = (last_row["queue_size"] - last_n) >= 1.0
        
        results.append({
            "saving": (dumb_gas_cost - oracle_gas_cost) / dumb_gas_cost if dumb_gas_cost > 0 else 0,
            "missed": missed,
            "avg_q": np.mean(q_size),
            "max_q": np.max(q_size),
            "holding_ratio": total_holding_cost / (oracle_gas_cost + total_holding_cost) if (oracle_gas_cost + total_holding_cost) > 0 else 0
        })

    return {
        "dist": dist,
        "avg_saving": np.mean([r["saving"] for r in results]),
        "miss_rate": np.mean([r["missed"] for r in results]),
        "avg_q": np.mean([r["avg_q"] for r in results]),
        "max_q": np.max([r["max_q"] for r in results]),
        "holding_ratio": np.mean([r["holding_ratio"] for r in results])
    }

if __name__ == "__main__":
    scenarios = [(0.1, 0.001), (0.3, 0.005), (0.5, 0.005)]
    for s, b in scenarios:
        try:
            print(f"\n[SCENARIO] Scale: {s}, Beta: {b} ...")
            res = run_sensitivity_test(s, b)
            print(f"--- FULL ANALYSIS for Scale={s}, Beta={b} ---")
            print(f" 💰 Tiết kiệm Gas (vs Dumb): {res['avg_saving']:.2%}")
            print(f" ⏱️ Hàng đợi TB (Avg Queue): {res['avg_q']:.1f} txs")
            print(f" 🌋 Hàng đợi Max (Max Queue): {res['max_q']:.1f} txs")
            print(f" ⚖️ Tỷ trọng phí phạt: {res['holding_ratio']:.2%}")
            print(f" 🚨 Tỷ lệ trễ (Deadline): {res['miss_rate']:.2%}")
            print(f" 📊 Phân bổ hành động:")
            for a, p in res['dist'].items():
                print(f"    Action {int(a)}: {p:.2%}")
        except Exception as e:
            print(f"Error testing Scenario {s}/{b}: {e}")
