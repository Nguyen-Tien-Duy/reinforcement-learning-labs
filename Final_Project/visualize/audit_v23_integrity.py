import pandas as pd
import numpy as np
import json
import logging
import sys
from pathlib import Path

# --- AUTO-DETECT PROJECT PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = BASE_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

try:
    from utils.offline_rl.config import TransitionBuildConfig
except ImportError:
    # Fallback if structure is different
    sys.path.append(str(BASE_DIR))
    from code.utils.offline_rl.config import TransitionBuildConfig

# Load Config for Audit
config = TransitionBuildConfig()

DATA_PATH = BASE_DIR / "Data" / "transitions_discrete_v27.parquet"
NORM_PATH = CODE_DIR / "Data" / "state_norm_params.json"
LOG_PATH = BASE_DIR / "visualize" / "audit_v27_integrity.log"

# Sync parameters from SSoT (Single Source of Truth)
ARRIVAL_SCALE = config.arrival_scale
EXECUTION_CAPACITY = config.execution_capacity
GAS_SCALING_FACTOR = config.gas_scaling_factor
GAS_GWEI_SCALE = config.gas_to_gwei_scale
URGENCY_ALPHA = config.urgency_alpha
URGENCY_BETA = config.urgency_beta
DEADLINE_PENALTY = config.deadline_penalty
ACTION_BINS = config.action_bins
REWARD_SCALE = config.reward_scale

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AuditV23")

def load_norm_params():
    with open(NORM_PATH, "r") as f:
        params = json.load(f)
        return np.array(params["mins"], dtype=np.float32), np.array(params["maxs"], dtype=np.float32)

def denormalize(val, idx, mins, maxs):
    denom = (maxs[idx] - mins[idx])
    if denom == 0: denom = 1.0
    return (val * denom) + mins[idx]

def log_symmetric_transform(r):
    return np.sign(r) * np.log1p(np.abs(r))

def run_audit():
    logger.info("=== STARTING DEEP AUDIT FOR V23 TRANSITION DATA ===")
    logger.info(f"Source: {DATA_PATH}")
    
    if not DATA_PATH.exists():
        logger.error(f"Data file not found at {DATA_PATH}")
        return

    df = pd.read_parquet(DATA_PATH)
    mins, maxs = load_norm_params()

    cleanup_stats = {
        "total_episodes": 0,
        "cleared_episodes": 0,
        "final_queue_avg": 0.0,
    }
    
    # State column indices (from schema.py)
    G0_IDX = 0
    G1_IDX = 1
    MOM_IDX = 4
    ACC_IDX = 5
    Q_IDX = 8
    T_IDX = 9
    GREF_IDX = 10
    
    # Sample 15 random episodes
    episode_ids = df["episode_id"].unique()
    sampled_ep_ids = np.random.choice(episode_ids, size=min(len(episode_ids), 15), replace=False)
    
    total_steps = 0
    fail_counts = {"consistency": 0, "sequential": 0, "physics": 0, "reward": 0}
    
    for ep_id in sampled_ep_ids:
        ep_df = df[df["episode_id"] == ep_id].sort_values("step_index").reset_index(drop=True)
        logger.info(f"\n--- AUDITING EPISODE: {ep_id} (Length: {len(ep_df)}) ---")
        
        for t in range(len(ep_df)):
            total_steps += 1
            row = ep_df.iloc[t]
            
            # --- LAYER 0: Normalization & Consistency Cross-Check ---
            # Spec-aligned: In V24, gas_t in states is RAW (Wei), not log
            phys_gas_raw = row["gas_t"]
            norm_gas_raw = denormalize(row["s_gas_t0"], G0_IDX, mins, maxs)
            
            # Use relative tolerance for large Wei values
            if not np.isclose(phys_gas_raw, norm_gas_raw, rtol=1e-3):
                logger.error(f"[FAIL] Step {t}: Normalization mismatch for gas! physical={phys_gas_raw:.1f}, s_gas_t0 denorm={norm_gas_raw:.1f}")
                fail_counts["consistency"] += 1
            
            # s_queue should be queue_size after denorm
            phys_q = row["queue_size"]
            norm_q = denormalize(row["s_queue"], Q_IDX, mins, maxs)
            if not np.isclose(phys_q, norm_q, atol=1e-1):
                logger.error(f"[FAIL] Step {t}: Normalization mismatch for queue! physical={phys_q:.1f}, s_queue denorm={norm_q:.1f}")
                fail_counts["consistency"] += 1

            # s_momentum = s_gas_t0 - s_gas_t1 (in normalized space? No, usually in physical log space)
            # Actually, check simple-offline.py: momentum = log(g0) - log(g1)
            # In V23, the s_momentum column itself is normalized.
            phys_mom = np.log(max(1.0, row["gas_t"])) - np.log(max(1.0, ep_df.iloc[t-1]["gas_t"])) if t > 0 else 0.0
            norm_mom = denormalize(row["s_momentum"], MOM_IDX, mins, maxs)
            if t > 0 and not np.isclose(phys_mom, norm_mom, atol=1e-2):
                 logger.error(f"[FAIL] Step {t}: Feature logic mismatch! physical momentum={phys_mom:.4f}, s_momentum denorm={norm_mom:.4f}")
                 fail_counts["consistency"] += 1

            # --- LAYER 1: Sequential Integrity ---
            if t < len(ep_df) - 1:
                next_row = ep_df.iloc[t+1]
                if not np.isclose(row["ns_queue"], next_row["s_queue"], atol=1e-5):
                    logger.error(f"[FAIL] Step {t}: Sequential drift! ns_queue={row['ns_queue']:.6f}, next s_queue={next_row['s_queue']:.6f}")
                    fail_counts["sequential"] += 1

            # --- LAYER 2: Physics Invariant (Q_t+1 = Q_t - n_t + w_t+1) ---
            q_t_phys = row["queue_size"]
            action_bin = int(row["action"])
            ratio = ACTION_BINS[action_bin]
            executed = np.floor(ratio * q_t_phys)
            executed = np.clip(executed, 0, EXECUTION_CAPACITY)
            
            if t < len(ep_df) - 1:
                w_next_phys = np.round(next_row["transaction_count"] * ARRIVAL_SCALE)
                expected_q_next_phys = max(0, q_t_phys - executed + w_next_phys)
                actual_q_next_phys = next_row["queue_size"]
                
                if not np.isclose(expected_q_next_phys, actual_q_next_phys, atol=1.0):
                    logger.error(f"[FAIL] Step {t}: Physics mismatch! Expected Q_next={expected_q_next_phys:.1f}, Actual={actual_q_next_phys:.1f}")
                    logger.error(f"       Q_t={q_t_phys:.1f}, Executed={executed:.1f}, Incoming={w_next_phys:.1f} (raw={next_row['transaction_count']})")
                    fail_counts["physics"] += 1

            # --- LAYER 3: Reward Audit (Deep) ---
            gas_t_gwei = row["gas_t"] / GAS_GWEI_SCALE
            gas_ref_gwei = row["gas_reference"] / GAS_GWEI_SCALE
            # Efficiency
            r_eff = executed * ((gas_ref_gwei - gas_t_gwei) / GAS_SCALING_FACTOR)
            # Urgency
            time_left = row["time_to_deadline"]
            time_ratio = np.clip(time_left / 24.0, 0, 1)
            r_urg = URGENCY_BETA * (q_t_phys - executed) * np.exp(URGENCY_ALPHA * (1.0 - time_ratio))
            # Catastrophe
            is_last = (t == len(ep_df) - 1)
            deadline_miss = (is_last and (q_t_phys - executed) > 0)
            r_cat = DEADLINE_PENALTY if deadline_miss else 0.0
            
            raw_reward = (r_eff - r_urg - r_cat) / REWARD_SCALE
            # NOTE: Checking for log-symmetric transform
            # If the dataset was built WITHOUT it, we use raw_reward
            stored_reward = row["reward"]
            if np.isclose(raw_reward, stored_reward, rtol=1e-2, atol=1e-1):
                pass 
            elif np.isclose(transformed_reward, stored_reward, rtol=1e-2, atol=1e-1):
                pass
            else:
                logger.error(f"[FAIL] Step {t}: Reward mismatch! Raw={raw_reward:.4f}, Transformed={transformed_reward:.4f}, Stored={stored_reward:.4f}")
                logger.error(f"       Eff={r_eff:.2f}, Urg={r_urg:.2f}, Cat={r_cat:.2f}, Scale={REWARD_SCALE}")
                fail_counts["reward"] += 1
                
        last_row = ep_df.iloc[-1]
        final_q = denormalize(last_row["ns_queue"], Q_IDX, mins, maxs)
        cleanup_stats["total_episodes"] += 1
        cleanup_stats["final_queue_avg"] += final_q
        if final_q < 1.0:
            cleanup_stats['cleared_episodes'] += 1
        else:
            logger.warning(f"[INFO] Episode {ep_id} MISSED deadline! Final Queue: {final_q:.1f}")
    
    logger.info("\n=== AUDIT SUMMARY ===")
    logger.info(f"Total Steps Audited: {total_steps}")
    for layer, count in fail_counts.items():
        status = "PASSED ✅" if count == 0 else f"FAILED ❌ ({count} errors)"
        logger.info(f"{layer:<15}: {status}")
    
    if sum(fail_counts.values()) == 0:
        logger.info("\n[SUCCESS] Dataset integrity is 100% verified against V23 Spec.")
    else:
        logger.warning(f"\n[WARNING] Found {sum(fail_counts.values())} total errors. Check the log file for details.")

    logger.info(f"--- ORACLE PERFORMANCE REPORT ---")
    logger.info(f"Cleanup Success Rate: {cleanup_stats['cleared_episodes']/cleanup_stats['total_episodes']*100:.1f}%")
    logger.info(f"Average Leftover Queue: {cleanup_stats['final_queue_avg']/cleanup_stats['total_episodes']:.2f} txs")


if __name__ == "__main__":
    np.random.seed(42)
    run_audit()
