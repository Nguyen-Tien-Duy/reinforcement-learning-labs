import pandas as pd
import numpy as np
from utils.offline_rl.config import TransitionBuildConfig
from utils.offline_rl.build_state_action import _derive_queue

def debug_step_by_step():
    path = "Final_Project/Data/transitions_v22_balanced.parquet"
    df = pd.read_parquet(path)
    config = TransitionBuildConfig()
    
    # Pick the first episode
    ep_id = df["episode_id"].unique()[0]
    ep_df = df[df["episode_id"] == ep_id].copy().reset_index(drop=True)
    
    print(f"--- Debugging Episode {ep_id} ---")
    
    # Manually re-run the logic of _derive_queue for step 0 and 1
    arrivals = np.round(ep_df["transaction_count"].to_numpy() * config.arrival_scale).astype(np.int64)
    bins = np.array(config.action_bins)
    actions = ep_df["action"].to_numpy(dtype=np.int64)
    exec_cap = 500
    
    q = arrivals[0]
    print(f"Step 0: Arrival={arrivals[0]}, Q_initial={q}, Action={actions[0]}")
    
    ratio = bins[actions[0]]
    executed = int(min(np.floor(ratio * q), exec_cap))
    print(f"Step 0 Execution: Ratio={ratio}, Executed={executed}")
    
    q_next_calc = q - executed + arrivals[1]
    print(f"Step 1 Predicted Q: {q} - {executed} + {arrivals[1]} = {q_next_calc}")
    print(f"Step 1 Actual Q in DF: {ep_df.loc[1, 'queue_size']}")
    
    if q_next_calc != ep_df.loc[1, 'queue_size']:
        print("\n🚨 DISCREPANCY DETECTED!")
        if ep_df.loc[1, 'queue_size'] == (q + arrivals[1]):
            print("❌ IMPACT: The execution volume was IGNORED during build (Executed treated as 0).")
    else:
        print("\n✅ Logic matches. Looking elsewhere...")

if __name__ == "__main__":
    debug_step_by_step()
