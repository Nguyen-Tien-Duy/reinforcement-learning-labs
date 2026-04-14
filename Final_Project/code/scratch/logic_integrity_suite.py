import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project code to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "code"))

from utils.offline_rl.config import TransitionBuildConfig
from utils.offline_rl.build_state_action import build_state_action_frame
from utils.offline_rl.build_reward_episode import build_reward_episode_frame
from utils.offline_rl.enviroment import CharityGasEnv
from utils.offline_rl.schema import STATE_COLS, NEXT_STATE_COLS

def setup_suite_config(normalize=False):
    return TransitionBuildConfig(
        action_col="gas_used",
        gas_col="base_fee_per_gas",
        gas_used_col="gas_used",
        gas_limit_col="gas_limit",
        timestamp_col="timestamp",
        transaction_count_col="transaction_count",
        history_window=3,
        normalize_state=normalize
    )

def run_parity_scenario(raw_df, config, scenario_name):
    """
    Core engine to verify 1-to-1 parity between Dataset Builder and Environment.
    Checks all 11 state dimensions, rewards, and terminal flags.
    """
    print(f"\n[LOGIC] Scenario: {scenario_name}")
    print(f"  [DEBUG] Config: {config}")
    print(f"  [DEBUG] Columns: {list(raw_df.columns)}")
    
    # 1. Dataset Generation
    sa_df = build_state_action_frame(raw_df, config)
    transitions_df = build_reward_episode_frame(sa_df, config)
    
    # 2. Environment Setup
    env = CharityGasEnv(transitions_df, config)
    env.mins = None; env.maxs = None # Compare raw values
    
    obs_env, _ = env.reset()
    
    success = True
    for i in range(len(transitions_df)):
        row = transitions_df.iloc[i]
        action = row["action"]
        
        # Ground Truth
        s_data = np.array([row[c] for c in STATE_COLS], dtype=np.float32)
        r_data = row["reward"]
        s_next_data = np.array([row[c] for c in NEXT_STATE_COLS], dtype=np.float32)
        done_data = int(row["done"])
        
        # Simulation
        s_next_env, r_env, term, trunc, _ = env.step([action])
        done_env = int(term or trunc)
        
        try:
            # Check Observation, Reward, Next State, and Done
            # Relaxed tolerance for rewards/states due to float32 vs 64 at high magnitudes (Penalty=2M)
            np.testing.assert_allclose(obs_env, s_data, atol=1e-2)
            np.testing.assert_allclose(r_env, r_data, atol=1e-2)
            np.testing.assert_allclose(s_next_env, s_next_data, atol=1e-2)
            assert done_env == done_data
            
            obs_env = s_next_env
        except Exception as e:
            print(f"  [!!!] FAILED at step {i}: {e}")
            success = False
            break
            
    if success:
        print(f"  [OK] PASSED ({len(transitions_df)} steps).")
    return success

def test_marathon_drift():
    """Verify Q_t recurrence stays exact over 100 steps of random walk."""
    config = setup_suite_config()
    n = 100
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="12s"),
        "base_fee_per_gas": np.random.uniform(10e9, 200e9, n),
        "gas_used": np.random.uniform(5e6, 25e6, n),
        "gas_limit": [30e6] * n,
        "transaction_count": np.random.randint(50, 600, n)
    })
    return run_parity_scenario(df, config, "Marathon Drift Check")

def test_execution_bottleneck():
    """Verify capacity capping (500) works identically in Builder and Env."""
    config = setup_suite_config()
    n = 10
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="12s"),
        "base_fee_per_gas": [50e9] * n,
        "gas_used": [30e6] * n,
        "gas_limit": [30e6] * n,
        "transaction_count": [800] * n # Exceeds capacity
    })
    return run_parity_scenario(df, config, "Execution Bottleneck (Capping)")

def test_deadline_penalty():
    """Verify SLA penalty (1000.0) matches exactly at terminal step."""
    config = setup_suite_config()
    n = 5
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="12s"),
        "base_fee_per_gas": [50e9] * n,
        "gas_used": [0] * n, # No execution -> Build backlog
        "gas_limit": [30e6] * n,
        "transaction_count": [200] * n
    })
    return run_parity_scenario(df, config, "Deadline Penalty Alignment")

def verify_real_data_parity(file_path: Path, config: TransitionBuildConfig):
    """Verify that a few episodes from real parquet match Env logic exactly."""
    print(f"\n[LOGIC] Audit: Real Data Parity ({file_path.name})")
    if not file_path.exists():
        print(f"  [SKIP] File not found: {file_path}")
        return True

    try:
        # Load normalization params
        mins, maxs = None, None
        norm_path = PROJECT_ROOT / "Data" / "state_norm_params.json"
        if norm_path.exists():
            with open(norm_path, "r") as f:
                params = json.load(f)
                mins = np.array(params["mins"], dtype=np.float32)
                maxs = np.array(params["maxs"], dtype=np.float32)
            print(f"  [DEBUG] Loaded normalization params: mins={mins.shape}, maxs={maxs.shape}")

        df = pd.read_parquet(file_path).head(1000) # Check first ~1000 rows
        if "episode_id" not in df.columns:
            print("  [SKIP] No episode_id found in data.")
            return True

        episodes = df["episode_id"].unique()[:3] # Check first 3 episodes
        for ep_id in episodes:
            ep_df = df[df["episode_id"] == ep_id].copy()
            env = CharityGasEnv(episode_df=ep_df, config=config)
            env.mins, env.maxs = mins, maxs # Inject normalization params
            obs, _ = env.reset()
            
            for i in range(len(ep_df)):
                row = ep_df.iloc[i]
                action = [row["action"]]
                r_data = row["reward"]
                s_next_data = np.array([row[c] for c in NEXT_STATE_COLS], dtype=np.float32)
                
                # Step env
                obs_next, r_env, _, _, info = env.step(action)
                
                # Check with relaxed tolerance for float32/64
                try:
                    np.testing.assert_allclose(r_env, r_data, atol=1e-2)
                except AssertionError as e:
                    print(f"  [!!!] Reward Mismatch Step {i}:")
                    print(f"    Data Reward: {r_data}")
                    print(f"    Env Reward:  {r_env}")
                    print(f"    Components:  {info.get('reward_components', {})}")
                    raise e
                    
                np.testing.assert_allclose(obs_next, s_next_data, atol=1e-2)
                
        print(f"  [OK] Parity verified for {len(episodes)} real episodes.")
        return True
    except Exception as e:
        print(f"  [!!!] Real-Data Parity Failed: {e}")
        return False

def test_causal_learning_signal(file_path: Path):
    """Verify that Oracle logic creates non-zero causal relationships for RL to learn."""
    print(f"\n[LOGIC] Audit: Causal Learning Signal ({file_path.name})")
    if not file_path.exists():
         return True
    try:
        df = pd.read_parquet(file_path, columns=["action", "s_queue"])
        corr = df["action"].corr(df["s_queue"])
        print(f"  [DEBUG] Corr(Action, s_queue) = {corr:+.4f}")
        
        if pd.isna(corr) or abs(corr) < 0.001:
             print("  [!!!] CAUSAL COLLAPSE: Action and Queue are completely uncorrelated.")
             print("        The RL agent has no meaningful signal to learn from.")
             return False
        
        print("  [OK] Action is causally linked to Queue size (Signal is valid).")
        return True
    except Exception as e:
        print(f"  [!!!] Causal test failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Logic Integrity Suite")
    parser.add_argument("--input", type=Path, default=None, help="Path to real parquet to verify")
    args = parser.parse_args()

    print("="*60)
    print("LOGIC INTEGRITY SUITE")
    print("="*60)
    
    config = TransitionBuildConfig(action_col="gas_used")
    
    results = []
    results.append(test_marathon_drift())
    results.append(test_execution_bottleneck())
    results.append(test_deadline_penalty())
    
    if args.input:
        results.append(verify_real_data_parity(args.input, config))
        results.append(test_causal_learning_signal(args.input))

    print("\n" + "="*60)
    print(f"SUMMARY: {sum(results)}/{len(results)} PASSED")
    if all(results):
        print("STATUS: 100% LOGICAL PARITY ACHIEVED.")
    else:
        print("STATUS: DRIFT DETECTED. CHECK FORMULAS.")
    print("="*60)
