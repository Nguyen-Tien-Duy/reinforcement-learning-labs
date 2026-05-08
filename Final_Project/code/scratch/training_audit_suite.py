import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sys
import json
import hashlib
from dataclasses import asdict
from pathlib import Path

# Add project code to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "code"))

from utils.offline_rl.config import TransitionBuildConfig
from utils.offline_rl.transition_builder import build_transitions
from utils.offline_rl.enviroment import CharityGasEnv
from utils.offline_rl.d3rlpy_adapter import build_d3rlpy_dataset
from utils.offline_rl.schema import STATE_COLS, NEXT_STATE_COLS

def audit_reward_magnitudes(config: TransitionBuildConfig | None = None):
    """Verify that gas savings and deadline penalties are balanced."""
    print("\n[TRAINING] Audit: Reward Magnitudes")
    if config is None:
        config = TransitionBuildConfig(action_col="gas_used")
    
    # Example: Saving 10 Gwei with 200 tx
    volume = 200
    gas_diff = 10 
    r_eff_scaled = (volume * config.C_mar * gas_diff) / config.reward_scale
    
    # Penalty
    penalty_scaled = config.deadline_penalty / config.reward_scale
    
    ratio = r_eff_scaled / (penalty_scaled + 1e-12)
    print(f"  Efficiency Gain (+10 Gwei) [200 tx]: {r_eff_scaled:.4f} pts")
    print(f"  Deadline Penalty (Scaled): -{penalty_scaled:.4f} pts")
    print(f"  Ratio (Saving vs Penalty): {ratio:.4f}x")
    
    # With 5B penalty: 30.0 / 5000.0 = 0.006x — penalty overwhelms savings.
    # Agent MUST execute before deadline or face catastrophic loss.
    if ratio > 1.0:
        print("  [WARNING] Penalty may be too weak — savings can exceed penalty!")
        return False
    print("  [OK] Penalty dominates savings — agent has strong incentive to meet deadlines.")
    return True

def verify_normalization_parity():
    """Verify that Env.reset/step with loaded params matches Dataset (Normalized)."""
    print("\n[TRAINING] Audit: Normalization Parity")
    config = TransitionBuildConfig(normalize_state=True, action_col="gas_used")
    
    # Mock data
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="12s"),
        "base_fee_per_gas": [100e9] * 5,
        "gas_used": [15e6] * 5,
        "gas_limit": [30e6] * 5,
        "transaction_count": [100] * 5
    })
    
    transitions = build_transitions(df, config)
    
    # Load params
    params_path = PROJECT_ROOT / "Data" / "state_norm_params.json"
    if params_path.exists():
        with open(params_path, "r") as f:
            params = json.load(f)
            mins, maxs = np.array(params["mins"]), np.array(params["maxs"])
    else:
        mins, maxs = None, None

    # Env Setup
    env = CharityGasEnv(transitions, config)
    env.mins, env.maxs = mins, maxs
    
    obs_env, _ = env.reset()
    obs_data = transitions.iloc[0][STATE_COLS].to_numpy(dtype=np.float32)
    
    try:
        np.testing.assert_allclose(obs_env, obs_data, atol=1e-5)
        print("  [OK] Normalization Sync: Env matches Dataset obs.")
        return True
    except Exception as e:
        print(f"  [!!!] Normalization audit failed: {e}")
        return False

def calculate_current_config_hash(config: TransitionBuildConfig) -> str:
    """Computes the hash of the current live configuration."""
    config_dict = asdict(config)
    config_json = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_json.encode("utf-8")).hexdigest()

def check_config_lock(file_path: Path, current_config: TransitionBuildConfig):
    """Verify if the dataset metadata matches the current project configuration."""
    print("\n[TRAINING] Audit: Config Lock (Metadata Fingerprint)")
    try:
        # Read parquet metadata using pyarrow
        meta = pq.read_metadata(file_path)
        custom_meta = meta.metadata or {}
        
        fingerprint_bytes = custom_meta.get(b"config_fingerprint")
        if not fingerprint_bytes:
            print("  [NOT FOUND] No config fingerprint in metadata. (Legacy dataset?)")
            return True
            
        stored_hash = fingerprint_bytes.decode("utf-8")
        current_hash = calculate_current_config_hash(current_config)
        
        print(f"  [DEBUG] Stored Hash:  {stored_hash}")
        print(f"  [DEBUG] Current Hash: {current_hash}")
        
        if stored_hash != current_hash:
            print("  [!!!] LOCK MISMATCH: Dataset was built with a DIFFERENT configuration than current code.")
            print("        Consider rebuilding the dataset or checking your config.py constants.")
            return False
        else:
            print("  [OK] Config Lock: Dataset matches current project configuration.")
            return True
    except Exception as e:
        print(f"  [!!!] Config lock audit failed: {e}")
        return False

def check_action_diversity(dataset: pd.DataFrame):
    """Analyze the distribution of actions to ensure enough exploration data."""
    print("\n[TRAINING] Audit: Action Diversity")
    try:
        actions = dataset["action"].to_numpy()
        
        # Calculate distribution
        boundary_low = np.mean(actions < 0.05) * 100
        boundary_high = np.mean(actions > 0.95) * 100
        mid_range = np.mean((actions >= 0.05) & (actions <= 0.95)) * 100
        
        print(f"  [DEBUG] Action Distribution:")
        print(f"    Low Boundary (<0.05): {boundary_low:.1f}%")
        print(f"    High Boundary (>0.95): {boundary_high:.1f}%")
        print(f"    Mid-Range (0.05-0.95): {mid_range:.1f}%")
        
        if mid_range < 5.0:
            print("  [!!!] WARNING: Very low action diversity. Agent may struggle with continuous control.")
        else:
            print("  [OK] Action distribution provides sufficient exploration signal.")
            
        return True
    except Exception as e:
        print(f"  [!!!] Action diversity audit failed: {e}")
        return False

def verify_d3rlpy_compatibility():
    """Verify that dataset builds without Terminal/Timeout collisions for d3rlpy v2.x."""
    print("\n[TRAINING] Audit: D3RLPy v2.x Compatibility")
    # Strict validation requires: state, action, reward, next_state, done, episode_id, timestamp
    df_dict = {
        "action": [0.5]*5, 
        "reward": [1.0]*5,
        "done": [1,0,0,0,1], 
        "truncated": [1,0,0,0,1],
        "episode_id": [1]*5,
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="12s")
    }
    for s_col, ns_col in zip(STATE_COLS, NEXT_STATE_COLS):
        df_dict[s_col]  = [0.0]*5
        df_dict[ns_col] = [0.0]*5
    df = pd.DataFrame(df_dict)
    try:
        dataset = build_d3rlpy_dataset(df)
        print(f"  [OK] Dataset built with {len(dataset.episodes)} episodes.")
        return True
    except Exception as e:
        print(f"  [!!!] D3RLPy Error: {e}")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training Audit Suite")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--action-col", type=str, default="gas_used",
        help="action_col used when building the dataset (must match build command). Default: gas_used")
    parser.add_argument("--no-normalize", action="store_true",
        help="Set if dataset was built with --disable-state-normalization")
    args = parser.parse_args()

    BASE_DIR = PROJECT_ROOT / "code"
    data_dir = PROJECT_ROOT / "Data"
    
    target = args.input
    if target is None:
        target = data_dir / "transitions_hardened_v2.parquet"
        if not target.exists():
            target = data_dir / "transitions_hardened_oracle.parquet"

    print("="*60)
    print("TRAINING AUDIT SUITE (Signal & Normalization)")
    print(f"Target: {target.name}")
    print("="*60)
    
    # Reconstruct exactly the same config used during build.
    # IMPORTANT: EVERY field in TransitionBuildConfig affects the fingerprint hash.
    # These values MUST match the CLI flags used in the build command
    # (see SIMULATED_FEE_USAGE.md §1 for the canonical build command).
    config = TransitionBuildConfig(
        action_col=args.action_col,
        normalize_state=not args.no_normalize,
        # --- Economic parameters (must match --build-from-raw flags) ---
        deadline_penalty=5000000000.0,
        urgency_beta=100.0,
        urgency_alpha=3.0,
        reward_scale=1000000.0,
        C_base=21000.0,
        C_mar=15000.0,
        gas_to_gwei_scale=1e9,
        execution_capacity=500.0,
        episode_hours=24,
        history_window=3,
    )
    print(f"[INFO] Auditing with: action_col='{args.action_col}', normalize_state={not args.no_normalize}")
    print(f"[INFO] deadline_penalty={config.deadline_penalty:.0f}, urgency_beta={config.urgency_beta}, reward_scale={config.reward_scale}")

    lock_result = check_config_lock(target, config)
    
    # 2. Load Data for further tests
    try:
        df = pd.read_parquet(target)
        results = [
            lock_result,
            audit_reward_magnitudes(config), 
            verify_normalization_parity(), 
            check_action_diversity(df),
            verify_d3rlpy_compatibility()
        ]
    except Exception as e:
        print(f"[!] Critical Error loading target: {e}")
        results = [False]
    
    print("\n" + "="*60)
    print(f"SUMMARY: {sum(results)}/{len(results)} PASSED")
    if all(results):
        print("STATUS: TRAINING PIPELINE IS FULLY SYNCHRONIZED.")
    else:
        print("STATUS: LOGIC HOLES OR CONFIG MISMATCH DETECTED.")
    print("="*60)
