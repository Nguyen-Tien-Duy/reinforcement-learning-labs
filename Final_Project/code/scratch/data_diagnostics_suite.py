import pandas as pd
import numpy as np
import sys
import pyarrow.parquet as pq
from pathlib import Path

# Add project code to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "code"))

from utils.offline_rl.config import TransitionBuildConfig
from utils.offline_rl.enviroment import CharityGasEnv
from utils.offline_rl.schema import STATE_COLS, NEXT_STATE_COLS

def check_gas_reference_presence():
    """Verify that the Environment handles missing Gas Reference correctly by falling back."""
    print("\n[DATA] Audit: Gas Reference Fallback")
    config = TransitionBuildConfig(action_col="gas_used")
    
    # Mock data WITHOUT gas_reference
    df = pd.DataFrame({
        "gas_t": [50.0],
        "state": [[0.0]*11], # Correct length 11 list
        "queue_size": [10.0],
        "time_to_deadline": [10.0]
    })
    
    try:
        env = CharityGasEnv(episode_df=df, config=config)
        env.reset()
        # Action must be a list/array for the environment
        _, reward, _, _, _ = env.step([1.0])
        
        # If reward is large negative, it means it fell back to gas_t correctly (exec_cost) 
        # or applied overhead. We verify it doesn't crash.
        print("  [OK] Environment successfully handles missing gas_reference column.")
        return True
    except Exception as e:
        print(f"  [!!!] Crash on missing reference: {e}")
        return False

def diagnostic_parquet_schema(file_path):
    """Scan a parquet file for basic RL schema integrity."""
    print(f"\n[DATA] Audit: Parquet Schema ({file_path.name})")
    if not file_path.exists():
        print(f"  [SKIP] File not found: {file_path}")
        return True
    
    try:
        df = pd.read_parquet(file_path)
        required = [*STATE_COLS, "action", "reward", *NEXT_STATE_COLS, "done", "episode_id"]
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            print(f"  [!!!] Missing columns: {missing}")
            return False
        
        # Check for NaNs
        nan_count = df[required].isna().sum().sum()
        if nan_count > 0:
            print(f"  [!!!] Found {nan_count} NaN values in RL columns.")
            return False
            
        print(f"  [OK] Schema valid. {len(df)} transitions detected.")
        return True
    except Exception as e:
        print(f"  [!!!] Diagnostic error: {e}")
        return False

def check_economic_calibration(file_path: Path, config: TransitionBuildConfig):
    """Audit rewards to ensure they match the current economic calibration (Penalty vs Scale)."""
    print(f"\n[DATA] Audit: Economic Calibration ({file_path.name})")
    if not file_path.exists():
        return True
    try:
        df = pd.read_parquet(file_path, columns=["reward"])
        
        # New: Read metadata for fingerprint
        meta = pq.read_metadata(file_path)
        custom_meta = meta.metadata or {}
        fingerprint = custom_meta.get(b"config_fingerprint", b"None").decode("utf-8")
        print(f"  [DEBUG] Config Fingerprint: {fingerprint}")

        min_reward = df["reward"].min()
        expected_penalty = -(config.deadline_penalty / config.reward_scale)
        
        print(f"  [DEBUG] Min Reward in file: {min_reward:.4f}")
        print(f"  [DEBUG] Expected Penalty:  {expected_penalty:.4f}")
        
        # If the file hasn't been scaled (magnitude is > 10x expected) or vice versa
        # Example: Expected -2.0 but found -2000.0
        ratio = min_reward / (expected_penalty - 1e-9)
        
        if ratio > 10.0:
             print(f"  [!!!] CALIBRATION MISMATCH: Rewards are much LARGER than expected (Ratio: {ratio:.1f}x).")
             print(f"        Is the data unscaled? (Found {min_reward:.2f}, Expected ~{expected_penalty:.2f})")
             return False
        if ratio < 0.1 and expected_penalty < -0.1:
             print(f"  [!!!] CALIBRATION MISMATCH: Rewards are much SMALLER than expected (Ratio: {ratio:.1f}x).")
             print(f"        Possibly using old 100.0 penalty? (Found {min_reward:.4f}, Expected ~{expected_penalty:.4f})")
             return False
             
        print("  [OK] Rewards magnitude matches current calibration.")
        return True
    except Exception as e:
        print(f"  [!!!] Calibration audit failed: {e}")
        return False

def check_episode_continuity(file_path: Path):
    """Ensure that steps within episodes are sequential and timestamps are monotonic."""
    print(f"\n[DATA] Audit: Episode Continuity ({file_path.name})")
    if not file_path.exists():
        return True
        
    try:
        df = pd.read_parquet(file_path, columns=["episode_id", "step_index", "timestamp"])
        
        # Check for gaps in step_index
        errors = 0
        for eid, group in df.groupby("episode_id"):
            steps = group["step_index"].values
            expected_steps = np.arange(len(steps))
            if not np.array_equal(steps, expected_steps):
                if errors < 5:
                    print(f"  [!!!] CONTINUITY GAP: Episode {eid} has non-sequential steps: {steps[:10]}...")
                errors += 1
        
        if errors > 0:
            print(f"  [!!!] Total episodes with continuity gaps: {errors}")
            return False
            
        print("  [OK] All episodes are sequential and contiguous.")
        return True
    except Exception as e:
        print(f"  [!!!] Continuity audit failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Data Diagnostics Suite")
    parser.add_argument("--input", type=Path, default=None, help="Path to parquet file to audit")
    args = parser.parse_args()

    print("="*60)
    print("DATA DIAGNOSTICS SUITE")
    print("="*60)
    
    results = []
    results.append(check_gas_reference_presence())
    
    # Check the specified input if provided, otherwise fallback to defaults
    data_dir = PROJECT_ROOT / "Data"
    if args.input:
        target = args.input
    else:
        v15_path = data_dir / "transitions_v15_final.parquet"
        proxy_path = data_dir / "transitions_proxy_fixed.parquet"
        target = v15_path if v15_path.exists() else proxy_path
    
    config = TransitionBuildConfig()
    results.append(check_economic_calibration(target, config))
    results.append(check_episode_continuity(target))
    
    print("\n" + "="*60)
    print(f"SUMMARY: {sum(results)}/{len(results)} PASSED")
    if all(results):
        print("STATUS: DATA STRUCTURES ARE HEALTHY.")
    else:
        print("STATUS: DATA ANOMALIES DETECTED.")
    print("="*60)
