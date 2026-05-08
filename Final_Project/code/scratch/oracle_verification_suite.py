import numpy as np
import sys
from pathlib import Path

# Add project code to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "code"))

from utils.offline_rl.oracle_builder import compute_god_view_trajectory
from utils.offline_rl.config import TransitionBuildConfig

def verify_oracle_economic_logic():
    """Verify that Oracle waits during high gas and executes during low gas."""
    print("\n[ORACLE] Audit: Economic Decision Logic")
    config = TransitionBuildConfig(action_col="gas_used")
    
    # Scenario: Gas is high (100) then drops to 10
    gas_prices = np.array([100, 100, 100, 10, 10, 10], dtype=np.float32)
    incoming = np.ones(6, dtype=np.float32)
    
    opt_a, _, _ = compute_god_view_trajectory(
        gas_prices=gas_prices,
        incoming_requests=incoming,
        Q_max=100,
        C_base=config.C_base,
        C_mar=config.C_mar,
        beta=config.urgency_beta,
        alpha=config.urgency_alpha,
        episode_hours=config.episode_hours,
        reward_scale=config.reward_scale,
        deadline_penalty=config.deadline_penalty
    )
    
    # Expectation: Actions at high gas (indices 0,1,2) should be 0 or small.
    # At low gas phase (indices 3+), the sum of actions should be significant.
    # The Oracle is smart and might wait until the very last block of the window.
    total_low_gas_action = np.sum(opt_a[3:])
    
    if opt_a[0] < 0.1 and total_low_gas_action > 0.5:
        print(f"  [OK] Oracle correctly prioritizes low gas phase (Total Action: {total_low_gas_action:.2f})")
        return True
    else:
        print(f"  [!!!] Oracle behavior suspicious. Actions: {opt_a}")
        return False

def verify_oracle_reward_alignment():
    """Verify that Oracle's decision matches Env's optimal choice at deadline."""
    print("\n[ORACLE] Audit: Reward Scale Alignment")
    config = TransitionBuildConfig(action_col="gas_used")
    
    gas_gwei = 100.0
    
    # Env View
    exec_reward = -(gas_gwei * config.C_base) / config.reward_scale
    miss_reward = -config.deadline_penalty / config.reward_scale
    env_choice = 1.0 if exec_reward > miss_reward else 0.0
    
    # Oracle View (1-step episode)
    opt_a, _, _ = compute_god_view_trajectory(
        gas_prices=np.array([gas_gwei]),
        incoming_requests=np.array([1]),
        Q_max=100, C_base=config.C_base, C_mar=config.C_mar,
        beta=config.urgency_beta, alpha=config.urgency_alpha,
        episode_hours=config.episode_hours,
        reward_scale=config.reward_scale,
        deadline_penalty=config.deadline_penalty
    )
    
    oracle_choice = 1.0 if opt_a[0] > 0.5 else 0.0
    
    if env_choice == oracle_choice:
        print("  [OK] Oracle and Env are perfectly aligned on Execute/Wait.")
        return True
    else:
        print(f"  [!!!] Alignment Failure: Env chose {env_choice}, Oracle chose {oracle_choice}.")
        return False

if __name__ == "__main__":
    print("="*60)
    print("ORACLE VERIFICATION SUITE")
    print("="*60)
    
    results = [verify_oracle_economic_logic(), verify_oracle_reward_alignment()]
    
    print("\n" + "="*60)
    print(f"SUMMARY: {sum(results)}/{len(results)} PASSED")
    if all(results):
        print("STATUS: HINDSIGHT ORACLE IS RELIABLE.")
    else:
        print("STATUS: ORACLE MISALIGNMENT DETECTED.")
    print("="*60)
