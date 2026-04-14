import numpy as np
import pandas as pd
import sys
import json
from pathlib import Path

# Add project code to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "code"))

from utils.offline_rl.config import TransitionBuildConfig
from utils.offline_rl.enviroment import CharityGasEnv

def load_normalization_params(data_dir: Path):
    params_path = data_dir / "state_norm_params.json"
    if params_path.exists():
        with open(params_path, "r") as f:
            params = json.load(f)
            return np.array(params["mins"]), np.array(params["maxs"])
    return None, None

def check_state_parity(file_path: Path, config: TransitionBuildConfig):
    """Verify that environment-generated states match the pre-built dataset."""
    print("\n[EVAL] Audit: State Parity (Training vs Simulation)")
    try:
        df = pd.read_parquet(file_path)
        
        # Pick the first episode with enough steps
        episodes = df.groupby("episode_id").size()
        target_ep = episodes[episodes > 50].index[0]
        ep_df = df[df["episode_id"] == target_ep].reset_index(drop=True)
        
        # Setup Env
        env = CharityGasEnv(episode_df=ep_df, config=config)
        
        # Load params if they exist
        mins, maxs = load_normalization_params(file_path.parent)
        env.mins = mins
        env.maxs = maxs
        
        state, _ = env.reset()
        
        errors = 0
        for i in range(len(ep_df)):
            stored_state = np.array(ep_df.iloc[i]["state"], dtype=np.float32)
            
            # Compare
            diff = np.abs(state - stored_state)
            if np.max(diff) > 1e-4:
                if errors < 3:
                    print(f"  [!!!] MISMATCH at Step {i}:")
                    print(f"    Simulated State: {state[8:11]}")
                    print(f"    Stored State:    {stored_state[8:11]}")
                    print(f"    Max Diff: {np.max(diff):.6f}")
                errors += 1
            
            # Advance using the action from data
            action = [ep_df.iloc[i]["action"]]
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
                
        if errors == 0:
            print(f"  [OK] State Parity: Simulated states match stored states for Episode {target_ep}.")
            return True
        else:
            print(f"  [!!!] State Parity FAILED: Found {errors} mismatching steps.")
            return False
            
    except Exception as e:
        print(f"  [!!!] State parity audit crashed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_cost_reporting(file_path: Path, config: TransitionBuildConfig):
    """Verify that simulated cost is accurately reported in info['cost']."""
    print("\n[EVAL] Audit: Cost Reporting Logic")
    try:
        df = pd.read_parquet(file_path)
        ep_df = df[df["episode_id"] == df["episode_id"].unique()[0]].reset_index(drop=True)
        
        env = CharityGasEnv(episode_df=ep_df, config=config)
        env.reset()
        
        # Take a step
        row = ep_df.iloc[0]
        action = [1.0] # Push for execution
        state, reward, term, trunc, info = env.step(action)
        
        reported_cost = info.get("cost")
        if reported_cost is None:
            print("  [!!!] MISSING COST: env.step() did not return 'cost' in info.")
            return False
            
        # Manual calculation
        gas_scale = config.gas_to_gwei_scale
        gas_t = float(row["gas_t"]) / gas_scale
        exec_vol = info["executed"]
        expected_cost = gas_t * exec_vol
        
        if abs(reported_cost - expected_cost) < 1e-7:
            print(f"  [OK] Cost Reporting: Reported={reported_cost:.4f}, Expected={expected_cost:.4f}")
            return True
        else:
            print(f"  [!!!] COST MISMATCH: Reported={reported_cost:.6f}, Expected={expected_cost:.6f}")
            return False
            
    except Exception as e:
        print(f"  [!!!] Cost audit crashed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluation Integrity Suite")
    parser.add_argument("--input", type=Path, default=None)
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "Data"
    target = args.input
    if target is None:
        target = data_dir / "transitions_hardened_v2.parquet"
        if not target.exists():
            target = data_dir / "transitions_hardened_oracle.parquet"

    print("="*60)
    print("EVALUATION INTEGRITY SUITE (Parity & Metrics)")
    print(f"Target: {target.name}")
    print("="*60)
    
    # We use the hardened config from the script
    config = TransitionBuildConfig(
        history_window=3,
        episode_hours=24,
        deadline_penalty=2000000.0,
        urgency_beta=100.0,
        reward_scale=1e6,
        C_base=21000.0,
        C_mar=15000.0,
        gas_to_gwei_scale=1e9
    )
    
    results = [
        check_state_parity(target, config),
        check_cost_reporting(target, config)
    ]
    
    print("\n" + "="*60)
    print(f"SUMMARY: {sum(results)}/{len(results)} PASSED")
    if all(results):
        print("STATUS: EVALUATION SIMULATOR IS FULLY SYNCHRONIZED.")
    else:
        print("STATUS: SIMULATION DRIFT DETECTED. CHECK ENVIROMENT.PY INDEXES/CONSTANTS.")
    print("="*60)
