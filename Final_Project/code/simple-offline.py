from __future__ import annotations

import json
import pickle
import hashlib
from dataclasses import asdict
from datetime import datetime
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from d3rlpy.algos import DiscreteCQLConfig, DiscreteBCQConfig
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.metrics import DiscreteActionMatchEvaluator
from utils.load_data import (
    TransitionBuildConfig,
    TransitionValidationError,
    build_transitions_from_parquet,
    build_d3rlpy_dataset,
    format_validation_report,
    load_transition_dataframe,
    validate_transition_dataframe,
)

from utils.offline_rl.schema import STATE_COLS
# Imports grouped above for Single Source of Truth handling

from utils.offline_rl.enviroment import CharityGasEnv

BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline RL parquet loader + transition schema validator"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=BASE_DIR.parent / "Data" / "data_2024-04-10_2026-04-10.parquet",
        help="Path to parquet file",
    )
    parser.add_argument(
        "--mode",
        choices=("strict", "lenient"),
        default="strict",
        help="Validation mode",
    )
    parser.add_argument(
        "--to-d3rlpy",
        action="store_true",
        help="Build a d3rlpy dataset object after validation passes",
    )
    parser.add_argument(
        "--build-from-raw",
        type=Path,
        default=None,
        help="Build transitions from raw parquet instead of loading prebuilt transition parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output parquet path for built transitions",
    )
    parser.add_argument(
        "--action-col",
        type=str,
        default=None,
        help="Raw column used to derive action (action=1 when value > threshold)",
    )
    parser.add_argument(
        "--queue-col",
        type=str,
        default=None,
        help="Optional raw queue column; if missing, queue proxy is derived",
    )
    # Toy arguments
    parser.add_argument(
        "--train-toy",
        action="store_true",
        help="Enable toy IQL training after validation"
    )
    parser.add_argument("--sample-size", type=int, default=1000000,
        help="Transitions to sample for toy training. Target 20-50x effective epochs: n_steps×batch/sample.")
    parser.add_argument(
        "--n-steps",
        type=int,
        default=20000,
        help="Number of training steps for toy IQL training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=36,
        help="Random seed for sampling and training",
    )
    # Evaluation arguments
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run offline evaluation metrics on a time-based holdout split",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to a trained d3rlpy model (.d3) for RL policy evaluation",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.2,
        help="Holdout ratio for time-based evaluation split (0, 1)",
    )
    parser.add_argument(
        "--baseline-gas-threshold",
        type=float,
        default=None,
        help="Gas threshold for execute-if-low baseline (if gas column exists)",
    )
    parser.add_argument(
        "--cvar-alpha",
        type=float,
        default=0.1,
        help="Alpha for CVaR metric (0, 1)",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help="Optional JSON path to save evaluation report",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=1.0,
        help="Reward scaling factor (divide raw rewards by this) for stable training",
    )
    parser.add_argument(
        "--eval-all",
        action="store_true",
        help="Evaluate all model checkpoints in a directory and generate a leaderboard",
    )
    parser.add_argument(
        "--eval-watch",
        action="store_true",
        help="Continuously watch for new model checkpoints and update the leaderboard live",
    )
    parser.add_argument(
        "--cql-alpha",
        type=float,
        default=1.0,
        help="CQL conservatism alpha (default 1.0, lower is less conservative)",
    )
    parser.add_argument(
        "--fee-gas-scale",
        type=float,
        default=1e9,
        help="Scale factor for gas unit conversion in fee simulation (default: 1e9, Wei->Gwei)",
    )
    parser.add_argument(
        "--execution-proxy-mode",
        choices=("queue", "unit"),
        default="queue",
        help="Execution volume proxy mode for simulated fee: queue uses queue_size/executed_volume_proxy, unit uses 1 per execute",
    )
    parser.add_argument(
        "--execution-capacity",
        type=float,
        default=TransitionBuildConfig.execution_capacity,
        help="Optional cap for execution proxy per step (applied when action=1)",
    )
    parser.add_argument(
        "--arrival-scale",
        type=float,
        default=0.5,
        help="Scale factor for incoming TX count (0.5 = use 50%% of raw tx_count as arrivals)",
    )
    parser.add_argument(
        "--no-eval-plots",
        action="store_true",
        help="Disable saving evaluation plots (metrics dashboard and pareto chart)",
    )
    # Transition build configuration arguments (Synced with TransitionBuildConfig defaults)
    parser.add_argument("--history-window", type=int, default=TransitionBuildConfig.history_window)
    parser.add_argument("--episode-hours", type=int, default=TransitionBuildConfig.episode_hours)
    parser.add_argument("--action-threshold", type=float, default=0.0)
    parser.add_argument("--deadline-penalty", type=float, default=TransitionBuildConfig.deadline_penalty)
    parser.add_argument("--urgency-beta", type=float, default=TransitionBuildConfig.urgency_beta)
    parser.add_argument("--urgency-alpha", type=float, default=TransitionBuildConfig.urgency_alpha)
    parser.add_argument("--C-base", type=float, default=TransitionBuildConfig.C_base if hasattr(TransitionBuildConfig, 'C_base') else 21000.0)
    parser.add_argument("--C-mar", type=float, default=TransitionBuildConfig.C_mar if hasattr(TransitionBuildConfig, 'C_mar') else 15000.0)
    parser.add_argument("--gas-reference-window", type=int, default=TransitionBuildConfig.gas_reference_window)
    parser.add_argument(
        "--disable-state-normalization",
        action="store_true",
        help="Disable min-max normalization when creating state vectors",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5000,
        help="Number of steps between model checkpoints during training",
    )
    parser.add_argument(
        "--use-oracle",
        action="store_true",
        help="Use Hindsight Oracle to override actions when building from raw",
    )
    parser.add_argument(
        "--expert-ratio",
        type=float,
        default=0.4,
        help="Tỷ lệ episodes Expert - 100%% Oracle DP (V28 Episode-Level)",
    )
    parser.add_argument(
        "--medium-ratio",
        type=float,
        default=0.3,
        help="Tỷ lệ episodes Medium - Epsilon-Greedy 70/30 (V28 Episode-Level)",
    )
    parser.add_argument(
        "--random-ratio",
        type=float,
        default=0.3,
        help="Tỷ lệ episodes Random - 100%% Random (V28 Episode-Level)",
    )
    parser.add_argument(
        "--skip-validation", 
        action="store_true", 
        help="Skip data validation for faster startup (use only if data is already known to be valid)"
    )
    
    parser.add_argument(
        "--limit-eval-episodes",
        type=int,
        default=50,
        help="Limit number of episodes to simulate during evaluation for speed (default: 50)"
    )
    
    parser.add_argument(
        "--oracle-only",
        action="store_true",
        help="Filter training dataset to only include Oracle transitions (policy_type == 1)"
    )
    
    return parser.parse_args()

def load_normalization_params(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Strictly load state normalization parameters. Raises error if missing."""
    params_path = data_dir / "state_norm_params.json"
    if not params_path.exists():
        raise FileNotFoundError(
            f"CRITICAL: Normalization file {params_path} not found! "
            "Policy requires strict normalization. Please build transitions first."
        )
    with open(params_path, "r") as f:
        params = json.load(f)
        return np.array(params["mins"], dtype=np.float32), np.array(params["maxs"], dtype=np.float32)

def calculate_config_hash(config: TransitionBuildConfig) -> str:
    """Computes a stable MD5 hash of the configuration to lock data to code."""
    config_dict = asdict(config)
    # Ensure stable ordering for hashing
    config_json = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_json.encode("utf-8")).hexdigest()

class BellmanLoggingCallback:
    """Logs detailed Bellman Equation diagnostics every N steps."""
    def __init__(self, dataset, interval=1000, gamma=0.99, reward_mean=0.0, reward_std=1.0, log_transform=True):
        self.dataset = dataset
        self.interval = interval
        self.gamma = gamma
        self.reward_mean = reward_mean
        self.reward_std = reward_std
        self.log_transform = log_transform

    def __call__(self, algo, epoch, total_step):
        if total_step % self.interval == 0:
            import numpy as np
            import random
            
            # Manual sampling from episodes for compatibility with d3rlpy v2.x
            # Use 64 samples for statistically meaningful Match Rate (was 5)
            DIAG_SAMPLE_SIZE = 64
            obs_list, act_list, rew_list, nobs_list, term_list = [], [], [], [], []
            sampled_count = 0
            while sampled_count < DIAG_SAMPLE_SIZE:
                ep = random.choice(self.dataset.episodes)
                if len(ep.observations) < 2:
                    continue # Skip very short episodes
                
                idx = random.randint(0, len(ep.observations) - 2)
                obs_list.append(ep.observations[idx])
                act_list.append(ep.actions[idx])
                rew_list.append(ep.rewards[idx])
                nobs_list.append(ep.observations[idx+1])
                # Handle d3rlpy v1/v2 compatibility safely
                is_terminal = (idx == len(ep.observations) - 2) and (getattr(ep, 'terminated', False) or getattr(ep, 'terminal', False))
                term_list.append(float(is_terminal))
                sampled_count += 1
            
            obs = np.array(obs_list)
            actions = np.array(act_list)
            rewards = np.array(rew_list).flatten()
            next_obs = np.array(nobs_list)
            terminals = np.array(term_list).flatten()

            # Predict current Q-values
            q_values = algo.predict_value(obs, actions)
            # Predict greedy actions for the next state
            next_actions = algo.predict(next_obs)
            # Predict value of the next state: max_a Q(s', a)
            next_q_values = algo.predict_value(next_obs, next_actions)
            
            # Bellman Target: R + gamma * (1-terminal) * Q(s', a')
            targets = rewards + self.gamma * (1.0 - terminals) * next_q_values
            
            # Oracle Alignment Match Rate (on full 64-sample batch)
            predicted_current_actions = algo.predict(obs)
            oracle_actions = actions.flatten()
            correct_matches = np.sum(predicted_current_actions == oracle_actions)
            match_rate = correct_matches / len(oracle_actions) * 100.0 if len(oracle_actions) > 0 else 0.0
            
            # Compute summary statistics
            td_errors = []
            for i in range(len(q_values)):
                log_reward = rewards[i]
                scaled_reward = (log_reward - self.reward_mean) / (self.reward_std + 1e-6)
                scaled_target = scaled_reward + self.gamma * (1.0 - terminals[i]) * next_q_values[i]
                td_errors.append(scaled_target - q_values[i])
            td_errors = np.array(td_errors)
            
            # Agent action distribution for this batch
            pred_unique, pred_counts = np.unique(predicted_current_actions, return_counts=True)
            pred_dist_str = " | ".join([f"A{int(a)}:{int(c)}" for a, c in zip(pred_unique, pred_counts)])
            
            # Oracle action distribution for this batch
            orac_unique, orac_counts = np.unique(oracle_actions, return_counts=True)
            orac_dist_str = " | ".join([f"A{int(a)}:{int(c)}" for a, c in zip(orac_unique, orac_counts)])
            
            print(f"\n[Step {total_step}] --- DIAGNOSTIC (n={DIAG_SAMPLE_SIZE}) ---")
            print(f"    [MATCH]  Oracle Alignment : {match_rate:.1f}%  ({correct_matches}/{len(oracle_actions)})")
            print(f"    [TD ERR] Mean: {td_errors.mean():.4f}  |  Std: {td_errors.std():.4f}  |  Max: {np.abs(td_errors).max():.4f}")
            print(f"    [Q-VAL]  Mean: {q_values.mean():.4f}  |  Min: {q_values.min():.4f}  |  Max: {q_values.max():.4f}")
            print(f"    [AGENT]  {pred_dist_str}")
            print(f"    [ORACLE] {orac_dist_str}")

            # Show detailed breakdown for only 3 samples (to keep logs readable)
            print("  --- AI BRAIN SCAN (Q-Values) ---")
            for i in range(min(3, len(q_values))):
                log_reward = rewards[i]
                physical_r = np.sign(log_reward) * (np.exp(np.abs(log_reward)) - 1.0)
                
                # Quét não AI để lấy toàn bộ điểm đánh giá cho 5 hành động
                all_qs = []
                for a_test in range(5):
                    dummy_action = np.full((1,), a_test, dtype=np.int32)
                    q_val = algo.predict_value(np.array([obs[i]]), dummy_action)[0]
                    all_qs.append(float(q_val))
                
                qs_str = " | ".join([f"Q{a}={q:.2f}" for a, q in enumerate(all_qs)])
                
                oracle_a = int(np.array(actions[i]).item())
                agent_a = int(np.array(predicted_current_actions[i]).item())
                
                print(f"  Sample {i}: Thầy dạy A={oracle_a}  |  Trò đoán A={agent_a}")
                print(f"       Brain: [{qs_str}]")
                print(f"       Math : R_phys={physical_r:.1f}  |  TD_err={td_errors[i]:.4f}")
            print("-------------------------------------------\n", flush=True)
            # [ANTI-OOM] Dọn rác tensor cache sau mỗi lần chẩn đoán
            import gc
            gc.collect()

def train_toy_iql(dataframe, args) -> int:
    # Step A: remove evaluation data from training data
    eval_df = _time_holdout_split(dataframe, args.eval_ratio)
    # Step B: get old data to train
    train_df = dataframe[~dataframe.index.isin(eval_df.index)].copy()
    
    if args.oracle_only:
        if "policy_type" in train_df.columns:
            n_before = len(train_df)
            train_df = train_df[train_df["policy_type"] == 1].copy()
            print(f"[+] Oracle-only mode: Filtered training data from {n_before} to {len(train_df)} transitions.")
        else:
            print("[!] Warning: --oracle-only requested but policy_type column missing. Using all data.")

    # Step C: sample n transitions from the training data
    n = min (args.sample_size, len(train_df))
    if n <= 0:
        print(f"Invalid sample size {args.sample_size}; must be > 0 and less than dataset size.")
        return 1
    
    # 2) Episode-Based Sampling: Preserve temporal trajectories
    unique_episodes = train_df["episode_id"].unique()
    # Estimate episodes needed for the requested sample_size
    avg_ep_len = train_df.groupby("episode_id").size().mean()
    n_episodes = max(1, int(n / avg_ep_len))
    
    import random
    random.seed(args.seed)
    sampled_ep_ids = random.sample(list(unique_episodes), min(len(unique_episodes), n_episodes))
    
    sampled_df = train_df[train_df["episode_id"].isin(sampled_ep_ids)].copy()
    sampled_df = sampled_df.sort_values(["episode_id", "timestamp"]).reset_index(drop=True)
    
    print(f"[+] Sampled {len(sampled_ep_ids)} episodes ({len(sampled_df)} transitions) for training.")

    # Convert to continues action
    sampled_df['action'] = sampled_df['action'].astype(np.int32)

    # 3) Log-Transformation cho TOÀN BỘ dataframe TRƯỚC khi chia tập
    # R_new = sign(R) * log(1 + |R|)
    # CRITICAL: Phải áp dụng trước khi chia train/eval để đảm bảo tính nhất quán
    print(f"[+] Applying Symmetric Log-Transformation to ALL Rewards...")
    raw_rewards_all = dataframe["reward"].to_numpy()
    log_rewards_all = np.sign(raw_rewards_all) * np.log1p(np.abs(raw_rewards_all))
    dataframe["reward"] = log_rewards_all
    
    # Tính Median/IQR trên log-space từ dữ liệu đã sample (training portion)
    sampled_log = sampled_df["reward"].to_numpy()  # sampled_df vẫn trỏ vào dataframe đã transform
    sampled_log = np.sign(sampled_log) * np.log1p(np.abs(sampled_log))
    median = float(np.median(sampled_log))
    q75, q25 = np.percentile(sampled_log, [75, 25])
    iqr = float(q75 - q25)
    if iqr == 0: iqr = 1.0

    print(f"[+] Log-Space Robust Scaling: Median={median:.4f}, IQR={iqr:.4f}")
    print(f"    Min (Log): {log_rewards_all.min():.4f}, Max (Log): {log_rewards_all.max():.4f}")

    # === INVERSE NORMALIZATION (Khôi phục vật lý cho Training - Fix Nén Kép) ===
    # Quy tắc: Khôi phục RAW trước khi đưa vào Model có Observation Scaler nội bộ.
    try:
        current_mins, current_maxs = load_normalization_params(BASE_DIR.parent / "Data")
        print(f"[*] Đang khôi phục vật lý cho {len(STATE_COLS)} đặc trưng trạng thái...")
        # LƯU Ý: Phải áp dụng trên TOÀN BỘ dataframe để cả tập Train và Eval đều đúng
        for i, col in enumerate(STATE_COLS):
            if col in dataframe.columns:
                # Safe Guard: Chỉ khôi phục nếu dữ liệu đang ở dạng [0, 1]
                if dataframe[col].max() <= 1.01 and dataframe[col].min() >= -1.01:
                    dataframe[col] = dataframe[col] * (current_maxs[i] - current_mins[i]) + current_mins[i]
                else:
                    print(f"  > Bỏ qua {col}: Có vẻ đã là giá trị vật lý (Max={dataframe[col].max():.2f})")
        print("✅ Khải thuật vật lý hoàn tất.")
    except Exception as e:
        print(f"⚠️ Cảnh báo Inverse Normalization thất bại: {e}. AI có thể bị nén kép!")

    # 4) Chia tập Train/Eval THEO THỜI GIAN (Đảm bảo đồng nhất với Cloud)
    # Quy tắc: Học quá khứ, thi tương lai (10% cuối cùng)
    unique_episodes = sorted(dataframe["episode_id"].unique())
    split_idx = int(len(unique_episodes) * 0.9)
    train_ids = unique_episodes[:split_idx]
    eval_ids = unique_episodes[split_idx:]
    
    train_df = dataframe[dataframe["episode_id"].isin(train_ids)].copy()
    eval_df = dataframe[dataframe["episode_id"].isin(eval_ids)].copy()
    
    print(f"[+] Multi-level split: {len(train_ids)} train vs {len(eval_ids)} eval episodes.")

    train_dataset = build_d3rlpy_dataset(train_df, mode=args.mode)
    eval_dataset = build_d3rlpy_dataset(eval_df, mode=args.mode)

    # [ANTI-OOM] Giải phóng DataFrame gốc và các bản sao sau khi đã convert sang d3rlpy dataset
    import gc
    del train_df
    gc.collect()
    print("[♻️] Đã dọn rác train_df để giải phóng RAM.")

    # Verify: Double-check reward and observation ranges
    sample_ep = train_dataset.episodes[0]
    print(f"[VERIFY] Train dataset reward sample: {sample_ep.rewards.min():.4f} to {sample_ep.rewards.max():.4f}")
    print(f"[VERIFY] Train dataset obs (Queue) sample: {sample_ep.observations[0][8]:.2f} (giá trị vật lý)")

    # 5) Configure model
    try:
        import d3rlpy
    except ImportError:
        print("d3rlpy is not available. Install d3rlpy in this environment to enable toy training.")
        return 1

    from d3rlpy.models import VectorEncoderFactory
    from d3rlpy.preprocessing import StandardRewardScaler, MinMaxObservationScaler
    # CPU Sweet-spot: [512, 512] (~270k params). Đủ thông minh nhưng không bóp nghẹt i5-11300H
    encoder = VectorEncoderFactory(hidden_units=[512, 512])

    # Strict Sync: Use the SAME physical mins/maxs for the model's scaler
    mins, maxs = load_normalization_params(BASE_DIR.parent / "Data")
    obs_scaler = MinMaxObservationScaler(minimum=mins, maximum=maxs)
    print(f"[+] Enforcing Strict Observation Scaler with physical bounds.")

    # === ALGORITHM SELECTION ===
    config = DiscreteBCQConfig(
        learning_rate=3e-4,
        batch_size=512,
        gamma=0.99,
        action_flexibility=0.5, # Tăng tự do (50%) để AI bớt bị gò ép vào Imitator
        encoder_factory=encoder,
        observation_scaler=obs_scaler,
        reward_scaler=StandardRewardScaler(mean=median, std=iqr),
    )

    # Vị trí số 2: DiscreteCQLConfig (Thuật toán cũ)
    # config = DiscreteCQLConfig(
    #     learning_rate=3e-4,
    #     batch_size=512,
    #     gamma=0.99,
    #     alpha=0.1,
    #     encoder_factory=encoder,
    #     observation_scaler=obs_scaler,
    #     reward_scaler=StandardRewardScaler(mean=median, std=iqr),
    # )

    # === DEEP DIAGNOSTICS: QUÉT NỘI NÃO AI TRƯỚC KHI TRAIN ===
    print("\n" + "🔍" * 20)
    print("--- KIỂM CHỨNG BỘ NÉN (SCALER SCAN) ---")
    sample_obs = np.array([train_dataset.episodes[0].observations[0]])
    sample_rew = float(train_dataset.episodes[0].rewards[0].item())
    
    # 1. Kiểm tra nén Observation (MinMax Vật lý)
    import torch
    sample_tensor = torch.tensor(sample_obs, dtype=torch.float32)
    obs_scaled_ai = obs_scaler.transform(sample_tensor).cpu().numpy()
    q_raw = float(sample_obs[0][8])
    q_scaled = float(obs_scaled_ai[0][8])
    print(f"[OBS] Queue Vật lý: {q_raw:.2f} -> AI thấy (Nén): {q_scaled:.6f}")
    
    # 2. Kiểm tra nén Reward (Standard Log-Space)
    # Công thức: (R - Median) / IQR
    rew_scaled_manual = (sample_rew - median) / iqr
    print(f"[REW] Reward Log: {sample_rew:.4f}")
    print(f"      Cấu hình: Median={median:.4f}, IQR={iqr:.4f}")
    print(f"      Tính thủ công: ({sample_rew:.4f} - {median:.4f}) / {iqr:.4f} = {rew_scaled_manual:.4f}")
    print("---  XÁC NHẬN: HỆ QUY CHIẾU ĐÃ ĐỒNG NHẤT SOTA ---")
    print("🔍" * 20 + "\n")

    algo = config.create(device="cpu")
    algo.build_with_dataset(train_dataset)

    # 6) Define Robust Evaluators
    class RobustActionMatchEvaluator:
        def __init__(self, eval_df):
            # Pre-cache states and oracle actions to make evaluation lightning fast
            self.states = eval_df[STATE_COLS].to_numpy(dtype=np.float32)
            self.oracle_actions = pd.to_numeric(eval_df["action"], errors="coerce").fillna(0).astype(np.int32).to_numpy()

        def __call__(self, algo, dataset):
            # Vectorized prediction - significantly faster than d3rlpy's default
            predicted_actions = algo.predict(self.states)
            match_rate = (predicted_actions == self.oracle_actions).mean()
            return float(match_rate)

    class SlaSimulationEvaluator:
        def __init__(self, eval_df, config, args, num_episodes=10):
            self.eval_df = eval_df
            self.transition_config = config
            self.args = args
            self.num_episodes = num_episodes

        def __call__(self, algo, dataset):
            # Chọn ngẫu nhiên N episodes từ tập eval_df để chạy simulator
            ep_ids = self.eval_df["episode_id"].unique()
            sampled_ids = np.random.choice(ep_ids, min(len(ep_ids), self.num_episodes), replace=False)
            sampled_df = self.eval_df[self.eval_df["episode_id"].isin(sampled_ids)]
            
            # Gọi hàm simulate_policy có sẵn
            res = simulate_policy(sampled_df, algo, self.transition_config, self.args)
            return res["simulated_deadline_miss_rate"]

    # 7) Periodic training with checkpoints and Deep Diagnostics
    out_dir = BASE_DIR.parent / "output" 
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting training for {args.n_steps} steps...")
    prefix = "DiscreteBCQ" if isinstance(config, DiscreteBCQConfig) else "DiscreteCQL"
    run_name = f"{prefix}_V6_{datetime.now().strftime('%Y%m%d_%H%M')}"

    # Diagnostics: Monitor convergence via TD-errors
    bellman_callback = BellmanLoggingCallback(
        train_dataset, 
        interval=1000, 
        gamma=args.gamma if hasattr(args, 'gamma') else 0.99,
        reward_mean=median,
        reward_std=iqr
    )

    algo.fit(
        train_dataset,
        n_steps=args.n_steps,           
        n_steps_per_epoch=args.save_interval, 
        show_progress=True,
        experiment_name=run_name,
        callback=bellman_callback,
        evaluators={
            "oracle_match": RobustActionMatchEvaluator(eval_df),
            "sla_miss_rate": SlaSimulationEvaluator(
                eval_df, 
                TransitionBuildConfig(**{k: v for k, v in vars(args).items() if k in TransitionBuildConfig.__dataclass_fields__}), 
                args, 
                num_episodes=10
            )
        }
    )

    print(f"Full training complete. Models saved in d3rlpy_logs/{run_name}")
    return 0

def simulate_policy(eval_df: pd.DataFrame, algo, config, args: argparse.Namespace) -> dict:
    # Batching by episode_id
    grouped = eval_df.groupby("episode_id")

    total_episodes = 0
    total_cost_sum = 0.0
    miss_count = 0
    
    episode_list = list(grouped)
    # Sampling is now pre-calculated in run_evaluation to save memory!
    # All episodes passed here will be evaluated.
    print(f"\n[+] Simulating policy on {len(episode_list)} episodes of evaluation data...")

    # Synchronization: Load normalization params for the Env
    mins, maxs = load_normalization_params(BASE_DIR.parent / "Data")

    for ep_id, ep_df in episode_list:
        total_episodes += 1
        
        # Create simulate environment for this episode with STRICT normalization
        env = CharityGasEnv(episode_df=ep_df, config=config, mins=mins, maxs=maxs)
        state, _ = env.reset()
        
        ep_cost = 0.0
        missed = False

        # Start simulation loop
        while True:
            # AI (algo) look at state and decide action
            # Predict returns a discrete action ID (scalar integer)
            action = int(algo.predict(np.array([state]))[0])

            # Send action to enviroment and get the consequence
            state, reward, terminated, truncated, info = env.step(action)

            ep_cost += info.get("cost", 0.0)

            if terminated or truncated:
                missed = info.get("deadline_miss", False)
                break
        
        total_cost_sum += ep_cost
        if missed:
            miss_count += 1
    
    # caculate final metrics
    miss_rate = miss_count / total_episodes if total_episodes > 0 else 0.0
    mean_cost = total_cost_sum / total_episodes if total_episodes > 0 else 0.0
    
    print(f"Completed {total_episodes} episodes.")
    print(f"Miss rate: {miss_rate:.4f}")
    print(f"Mean cost: {mean_cost:.4f}")
    
    # [ANTI-OOM] Dọn rác Env objects sau khi simulate xong
    import gc
    gc.collect()
    
    return {
        "simulated_cost_per_episode": float(mean_cost),
        "simulated_deadline_miss_rate": float(miss_rate),
        "simulated_total_episodes": int(total_episodes)
    }

def _load_model_iql(model_path: Path, eval_df: pd.DataFrame, mode: str):
    import d3rlpy
    import pickle
    model_path_str = str(model_path)
    algo = None
    try:
        algo = d3rlpy.load_learnable(model_path_str, device="cpu")
    except (pickle.UnpicklingError, ValueError, RuntimeError, OSError, AttributeError, TypeError):
        try:
            # Fallback for weight-only files
            bootstrap_n = min(2048, len(eval_df))
            bootstrap_df = eval_df.head(bootstrap_n).copy()
            bootstrap_dataset = build_d3rlpy_dataset(bootstrap_df, mode=mode)
            from d3rlpy.algos import DiscreteCQLConfig, DiscreteBCQConfig
            from d3rlpy.models import VectorEncoderFactory
            from d3rlpy.preprocessing import MinMaxObservationScaler
            
            # Load normalization stats from the central source of truth
            mins, maxs = load_normalization_params(BASE_DIR.parent / "Data")
            
            # --- EVALUATION ALGORITHM SELECTION ---
            # 1. Evaluate BCQ Model
            algo = DiscreteBCQConfig(
                learning_rate=3e-4,
                batch_size=256,
                gamma=0.99,
                action_flexibility=0.3,
                encoder_factory=VectorEncoderFactory(hidden_units=[256, 256]),
                observation_scaler=MinMaxObservationScaler(minimum=mins, maximum=maxs), 
            ).create(device="cpu")
            
            # 2. Evaluate CQL Model (Uncomment if needed)
            # algo = DiscreteCQLConfig(
            #     learning_rate=3e-4,
            #     batch_size=256,
            #     gamma=0.99,
            #     alpha=0.1,
            #     encoder_factory=VectorEncoderFactory(hidden_units=[256, 256]),
            #     observation_scaler=MinMaxObservationScaler(minimum=mins, maximum=maxs), 
            # ).create(device="cpu")
            algo.build_with_dataset(bootstrap_dataset)
            algo.load_model(model_path_str)
        except Exception as e:
            print(f"Failed to load model {model_path}: {e}")
            return None
    return algo

def _eval_single_model(m_path, eval_df, sim_config, args):
    """Worker function for evaluating a single model in parallel."""
    print(f"[Worker] Evaluating {m_path.name}...", flush=True)
    algo = _load_model_iql(m_path, eval_df, args.mode)
    if algo is None: 
        return None
    sim_res = simulate_policy(eval_df, algo, sim_config, args)
    return {
        "model": m_path.name,
        "deadline_miss_rate": sim_res["simulated_deadline_miss_rate"],
        "cost_per_episode": sim_res["simulated_cost_per_episode"],
    }

def run_evaluation(dataframe: pd.DataFrame, args: argparse.Namespace) -> int:
    if not (0.0 < args.eval_ratio < 1.0):
        print("eval_ratio must be between 0 and 1 (exclusive).")
        return 1

    if not (0.0 < args.cvar_alpha < 1.0):
        print("cvar_alpha must be between 0 and 1 (exclusive).")
        return 1

    if args.fee_gas_scale <= 0:
        print("fee_gas_scale must be > 0.")
        return 1

    if args.execution_capacity is not None and args.execution_capacity <= 0:
        print("execution_capacity must be > 0 when provided.")
        return 1

    eval_df = _time_holdout_split(dataframe, args.eval_ratio)
    
    # IMMEDIATE MEMORY OPTIMIZATION 1: Delete massive full dataframe and Force GC
    import gc
    del dataframe
    gc.collect()
    
    if eval_df.empty:
        print("Evaluation split is empty. Increase dataset size or reduce eval_ratio.")
        return 1

    # IMMEDIATE MEMORY OPTIMIZATION 2: Pre-sample episodes before IPC pickling!
    if args.limit_eval_episodes and args.limit_eval_episodes > 0:
        episode_ids = eval_df["episode_id"].unique()
        if len(episode_ids) > args.limit_eval_episodes:
            import random
            random.seed(args.seed)
            sampled_eps = random.sample(list(episode_ids), args.limit_eval_episodes)
            eval_df = eval_df[eval_df["episode_id"].isin(sampled_eps)].copy()
            # Reclaim RAM from the unsampled rows
            gc.collect()
            print(f"[-] Discarded unsampled rows. Memory-optimized eval_df has {len(eval_df)} rows.")

    policies: dict[str, np.ndarray] = {}
    logged_action = pd.to_numeric(eval_df["action"], errors="coerce").fillna(0).astype(np.int32).to_numpy()
    policies["logged_policy"] = logged_action
    policies["execute_now"] = np.ones(len(eval_df), dtype=np.float32)

    gas_column = _pick_gas_column(eval_df)
    if gas_column is not None:
        gas_values = pd.to_numeric(eval_df[gas_column], errors="coerce").fillna(0.0)
        threshold = args.baseline_gas_threshold
        if threshold is None:
            threshold = float(gas_values.median())
        threshold_action = (gas_values <= threshold).astype(np.int64).to_numpy()
        policies[f"threshold_policy(gas<={threshold:.6g})"] = threshold_action

    if args.evaluate and (args.eval_all or args.eval_watch):
        if args.model_path is None or not args.model_path.is_dir():
            print("To evaluate all/watch models, --model-path must be a directory containing .d3 files")
            return 1
            
        import re
        import time
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]

        model_files = sorted(list(args.model_path.glob("*.d3")), key=lambda x: natural_sort_key(x.name))
        
        print(f"[+] Setting up leaderboard evaluation.")
        leaderboard_data = []
        evaluated_files = set()

        sim_config = TransitionBuildConfig(
            action_col=args.action_col, 
            queue_col=args.queue_col,
            history_window=args.history_window,
            episode_hours=args.episode_hours,
            action_threshold=args.action_threshold,
            deadline_penalty=args.deadline_penalty,
            queue_penalty=0.0,
            gas_reference_window=args.gas_reference_window,
            normalize_state= not args.disable_state_normalization,
            urgency_alpha=args.urgency_alpha,
            urgency_beta=args.urgency_beta,
            reward_scale=args.reward_scale,
            C_base=args.C_base,
            C_mar=args.C_mar,
            gas_to_gwei_scale=args.fee_gas_scale,
            execution_capacity=args.execution_capacity if args.execution_capacity is not None else 500.0,
            arrival_scale=args.arrival_scale,
        )

        import concurrent.futures
        import multiprocessing

        max_workers = max(1, multiprocessing.cpu_count() - 5)  # Leave 1 core for OS
        mp_context = multiprocessing.get_context("spawn")
        
        print(f"\n[+] Starting parallel evaluation with {max_workers} processes (using SPAWN to save RAM)...", flush=True)

        while True:
            # Refresh model list
            model_files = sorted(list(args.model_path.glob("*.d3")), key=lambda x: natural_sort_key(x.name))
            new_files = [m for m in model_files if m not in evaluated_files]
            
            if not new_files:
                if args.eval_watch:
                    time.sleep(15)
                    continue
                else: # eval-all finished
                    break
                    
            print(f"\n[+] Found {len(new_files)} NEW models. Simulating...")
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
                futures = {
                    executor.submit(_eval_single_model, m_path, eval_df, sim_config, args): m_path 
                    for m_path in new_files
                }
                
                for future in concurrent.futures.as_completed(futures):
                    m_path = futures[future]
                    try:
                        res = future.result()
                        if res is not None:
                            leaderboard_data.append(res)
                        evaluated_files.add(m_path)
                    except Exception as exc:
                        print(f"Error evaluating {m_path.name}: {exc}")

            if leaderboard_data:
                lb_df = pd.DataFrame(leaderboard_data)
                # Sort: Primary = Miss Rate (lower better), Secondary = Cost (lower better)
                lb_df = lb_df.sort_values(["deadline_miss_rate", "cost_per_episode"]).reset_index(drop=True)
                lb_df["rank"] = lb_df.index + 1
                
                print(f"\n=== LIVE MODEL LEADERBOARD (Evaluated: {len(leaderboard_data)}) ===")
                print(lb_df.to_string(index=False))
                
                run_id = datetime.now().strftime("%Y%m%d_%H%M")
                lb_csv = BASE_DIR.parent / "result" / f"leaderboard_{run_id}.csv"
                lb_csv.parent.mkdir(parents=True, exist_ok=True)
                lb_df.to_csv(lb_csv, index=False)
            
            if not args.eval_watch:
                print(f"\n[+] Leaderboard saved to {lb_csv}")
                break

        return 0

    if args.model_path is not None:
        if not args.model_path.exists():
            print(f"Model file not found: {args.model_path}")
            return 1
        try:
            import d3rlpy
        except ImportError as exc:
            print(f"d3rlpy import failed for evaluation: {exc}")
            return 1

        from utils.offline_rl.schema import STATE_COLS
        states = eval_df[STATE_COLS].to_numpy(dtype=np.float32)
        algo = _load_model_iql(args.model_path, eval_df, args.mode)
        if algo is None: return 1

        predicted_action = np.asarray(algo.predict(states)).reshape(-1).astype(np.int32)
        policies["rl_policy"] = predicted_action

        sim_config = TransitionBuildConfig(
            action_col=args.action_col, 
            queue_col=args.queue_col,
            history_window=args.history_window,
            episode_hours=args.episode_hours,
            action_threshold=args.action_threshold,
            deadline_penalty=args.deadline_penalty,
            queue_penalty=0.0,
            gas_reference_window=args.gas_reference_window,
            normalize_state= not args.disable_state_normalization,
            urgency_alpha=args.urgency_alpha,
            urgency_beta=args.urgency_beta,
            reward_scale=args.reward_scale,
            C_base=args.C_base,
            C_mar=args.C_mar,
            gas_to_gwei_scale=args.fee_gas_scale,
            execution_capacity=args.execution_capacity if args.execution_capacity is not None else 500.0,
            arrival_scale=args.arrival_scale,
        )
        # Run simulation with the trained model
        sim_result = simulate_policy(eval_df, algo, sim_config, args)

    results: dict[str, dict[str, float | int | str | None]] = {}
    for policy_name, action_hat in policies.items():
        results[policy_name] = _compute_metrics_for_policy(
            eval_df,
            action_hat,
            cvar_alpha=args.cvar_alpha,
            gas_column=gas_column,
            fee_gas_scale=args.fee_gas_scale,
            execution_proxy_mode=args.execution_proxy_mode,
            execution_capacity=args.execution_capacity,
        )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows_total": int(len(dataframe)),
        "rows_eval": int(len(eval_df)),
        "eval_ratio": float(args.eval_ratio),
        "available_policies": list(results.keys()),
        "metrics": results,
        "simulated_results": sim_result,
        "skipped_estimators": {
            "WIS": "Skipped: behavior policy probabilities/log-propensity are unavailable.",
            "DR": "Skipped: fitted Q and behavior propensity models are not available in current pipeline.",
            "FQE": "Skipped: dedicated FQE training/evaluation pipeline is not implemented yet.",
        },
    }

    _print_evaluation_report(report)

    output_path = args.report_output
    if output_path is None:
        out_dir = BASE_DIR.parent / "result"
        out_dir.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = out_dir / f"evaluation_{run_id}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Evaluation report saved to {output_path}")

    _save_evaluation_plots(report, output_path, disabled=args.no_eval_plots)

    return 0


def _time_holdout_split(dataframe: pd.DataFrame, eval_ratio: float) -> pd.DataFrame:
    temp = dataframe.copy()
    parsed_ts = pd.to_datetime(temp["timestamp"], utc=True, errors="coerce")

    if parsed_ts.isna().all():
        temp["_order"] = np.arange(len(temp))
        temp = temp.sort_values("_order")
    else:
        temp["_parsed_ts"] = parsed_ts
        temp = temp.sort_values("_parsed_ts", na_position="last")

    split_idx = int((1.0 - eval_ratio) * len(temp))
    eval_df = temp.iloc[split_idx:].copy()
    for col in ["_parsed_ts", "_order"]:
        if col in eval_df.columns:
            eval_df.drop(columns=[col], inplace=True)
    return eval_df.reset_index(drop=True)


def _pick_gas_column(dataframe: pd.DataFrame) -> str | None:
    for col in ["gas_t", "base_fee_per_gas", "gas"]:
        if col in dataframe.columns:
            return col
    return None


def _compute_metrics_for_policy(
    eval_df: pd.DataFrame,
    action_hat: np.ndarray,
    *,
    cvar_alpha: float,
    gas_column: str | None,
    fee_gas_scale: float,
    execution_proxy_mode: str,
    execution_capacity: float | None,
) -> dict[str, float | int | str | None]:
    logged_action = pd.to_numeric(eval_df["action"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    reward = pd.to_numeric(eval_df["reward"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    episode_id = eval_df["episode_id"]

    # For continuous actions, exact coverage might be zero, but we keep the logic
    # or calculate distance-based match later. For now, matching the types.
    action_hat = np.asarray(action_hat).reshape(-1).astype(np.int32)
    if len(action_hat) != len(eval_df):
        raise ValueError("Predicted action length does not match evaluation dataframe length.")

    matched = np.isclose(action_hat, logged_action, atol=1e-3)
    coverage = float(matched.mean())

    expected_return_matched = None
    if matched.any():
        expected_return_matched = float(reward[matched].mean())

    action_rate = float((action_hat > 0).mean())

    exec_count_per_episode = pd.Series((action_hat == 1).astype(np.int8)).groupby(episode_id, dropna=False).sum()
    deadline_miss_rate = float((exec_count_per_episode == 0).mean())

    total_cost_per_episode = None
    total_cost_sum = None
    execution_proxy_per_episode = None
    execute_count_per_episode = float(exec_count_per_episode.mean())
    if gas_column is not None:
        gas_value = pd.to_numeric(eval_df[gas_column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        executed_volume_proxy = _derive_execution_proxy(
            eval_df,
            action_hat,
            mode=execution_proxy_mode,
            capacity=execution_capacity,
        )
        gas_converted = gas_value / fee_gas_scale
        step_cost = gas_converted * executed_volume_proxy
        episode_cost = pd.Series(step_cost).groupby(episode_id, dropna=False).sum()
        total_cost_per_episode = float(episode_cost.mean())
        total_cost_sum = float(episode_cost.sum())

        episode_exec_proxy = pd.Series(executed_volume_proxy).groupby(episode_id, dropna=False).sum()
        execution_proxy_per_episode = float(episode_exec_proxy.mean())

    adjusted_return = None
    if expected_return_matched is not None:
        adjusted_return = float(coverage * expected_return_matched)

    matched_episode_return = pd.Series(np.where(matched, reward, 0.0)).groupby(episode_id, dropna=False).sum()
    risk_variance_episode = float(matched_episode_return.var(ddof=0))

    cvar_episode = None
    if len(matched_episode_return) > 0:
        q = float(np.quantile(matched_episode_return.to_numpy(dtype=np.float64), cvar_alpha))
        tail = matched_episode_return[matched_episode_return <= q]
        if len(tail) > 0:
            cvar_episode = float(tail.mean())

    return {
        "coverage": coverage,
        "expected_return_matched": expected_return_matched,
        "action_rate": action_rate,
        "deadline_miss_rate": deadline_miss_rate,
        "total_cost_per_episode": total_cost_per_episode,
        "total_cost_sum": total_cost_sum,
        "execution_proxy_per_episode": execution_proxy_per_episode,
        "fee_gas_scale": float(fee_gas_scale),
        "execution_proxy_mode": execution_proxy_mode,
        "execute_count_per_episode_proxy": execute_count_per_episode,
        "adjusted_return": adjusted_return,
        "reward_variance_episode": risk_variance_episode,
        f"cvar_episode_alpha_{cvar_alpha}": cvar_episode,
    }


def _derive_execution_proxy(
    eval_df: pd.DataFrame,
    action_hat: np.ndarray,
    *,
    mode: str,
    capacity: float | None,
) -> np.ndarray:
    action_mask = (action_hat == 1).astype(np.float64)

    if mode == "unit":
        executed = action_mask
    else:
        if "executed_volume_proxy" in eval_df.columns:
            base_volume = pd.to_numeric(eval_df["executed_volume_proxy"], errors="coerce").fillna(0.0)
        elif "queue_size" in eval_df.columns:
            base_volume = pd.to_numeric(eval_df["queue_size"], errors="coerce").fillna(0.0)
        else:
            base_volume = pd.Series(np.ones(len(eval_df), dtype=np.float64))

        executed = action_mask * base_volume.to_numpy(dtype=np.float64)

    if capacity is not None:
        executed = np.minimum(executed, capacity)

    executed = np.maximum(executed, 0.0)
    return executed


def _print_evaluation_report(report: dict[str, object]) -> None:
    print("=== Evaluation Summary ===")
    print(f"rows_total={report['rows_total']}")
    print(f"rows_eval={report['rows_eval']}")
    print(f"eval_ratio={report['eval_ratio']}")

    metrics = report["metrics"]
    if isinstance(metrics, dict):
        for policy_name, metric in metrics.items():
            print(f"--- {policy_name} ---")
            if isinstance(metric, dict):
                for key, value in metric.items():
                    print(f"{key}: {value}")

    skipped = report.get("skipped_estimators")
    if isinstance(skipped, dict):
        print("--- Skipped Estimators ---")
        for key, reason in skipped.items():
            print(f"{key}: {reason}")


def _save_evaluation_plots(report: dict[str, object], report_path: Path, *, disabled: bool) -> None:
    if disabled:
        print("Evaluation plots are disabled by --no-eval-plots.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available, skip saving evaluation plots.")
        return

    metrics_obj = report.get("metrics")
    if not isinstance(metrics_obj, dict) or len(metrics_obj) == 0:
        print("No policy metrics available, skip plotting.")
        return

    policies = list(metrics_obj.keys())

    def extract_metric(metric_name: str) -> np.ndarray:
        values: list[float] = []
        for policy in policies:
            policy_metrics = metrics_obj.get(policy)
            value = None
            if isinstance(policy_metrics, dict):
                value = policy_metrics.get(metric_name)

            if isinstance(value, (int, float)) and np.isfinite(value):
                values.append(float(value))
            else:
                values.append(np.nan)

        return np.asarray(values, dtype=np.float64)

    metric_specs = [
        ("coverage", "Coverage"),
        ("expected_return_matched", "Expected Return (Matched)"),
        ("adjusted_return", "Adjusted Return"),
        ("action_rate", "Action Rate"),
        ("deadline_miss_rate", "Deadline Miss Rate"),
        ("total_cost_per_episode", "Total Cost Per Episode"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()
    x = np.arange(len(policies))

    for ax, (metric_key, title) in zip(axes_flat, metric_specs):
        values = extract_metric(metric_key)
        ax.bar(x, np.nan_to_num(values, nan=0.0))
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    metrics_plot_path = report_path.with_name(f"{report_path.stem}_metrics.png")
    plt.savefig(metrics_plot_path, dpi=180)
    plt.close(fig)
    print(f"Evaluation metrics plot saved to {metrics_plot_path}")

    cost = extract_metric("total_cost_per_episode")
    cost_label = "Total Cost Per Episode"
    if not np.isfinite(cost).any():
        cost = extract_metric("execute_count_per_episode_proxy")
        cost_label = "Execute Count Per Episode (Proxy)"

    miss = extract_metric("deadline_miss_rate")
    valid = np.isfinite(cost) & np.isfinite(miss)
    if valid.any():
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.scatter(cost[valid], miss[valid], s=60)
        for idx, policy in enumerate(policies):
            if valid[idx]:
                ax2.annotate(policy, (cost[idx], miss[idx]), xytext=(5, 5), textcoords="offset points")

        ax2.set_title("Policy Pareto View")
        ax2.set_xlabel(cost_label)
        ax2.set_ylabel("Deadline Miss Rate")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()

        pareto_plot_path = report_path.with_name(f"{report_path.stem}_pareto.png")
        plt.savefig(pareto_plot_path, dpi=180)
        plt.close(fig2)
        print(f"Evaluation pareto plot saved to {pareto_plot_path}")
    else:
        print("No valid cost/miss values for pareto plot, skipped.")


def main() -> int:
    args = parse_args()

    # V22: Unified configuration from config.py SSoT
    # Only override columns if explicitly provided via CLI
    conf_kwargs = {"normalize_state": not args.disable_state_normalization}
    if args.action_col: conf_kwargs["action_col"] = args.action_col
    if args.queue_col:  conf_kwargs["queue_col"] = args.queue_col
    
    config = TransitionBuildConfig(**conf_kwargs)

    config_hash = calculate_config_hash(config)

    print("\n========================================")
    print("TRANSITION BUILD/TRAIN CONFIGURATION")
    for k, v in asdict(config).items():
        print(f"  {k:<20}: {v}")
    print("========================================\n")
    print(f"[+] Config Fingerprint: {config_hash}")

    if args.build_from_raw is not None:
        if not args.build_from_raw.exists():
            print(f"Raw parquet not found: {args.build_from_raw}")
            return 1

        try:
            dataframe = build_transitions_from_parquet(
                args.build_from_raw,
                config,
                use_oracle=args.use_oracle,
                expert_ratio=args.expert_ratio,
                medium_ratio=args.medium_ratio,
                random_ratio=args.random_ratio,
                config_hash=config_hash,
            )
            
            # FINAL Normalization Sweep: Must occur BEFORE saving the file
            if config.normalize_state:
                from utils.offline_rl.schema import STATE_COLS, NEXT_STATE_COLS
                
                state_matrix = dataframe[STATE_COLS].to_numpy(dtype=np.float32)
                mins = state_matrix.min(axis=0)
                maxs = state_matrix.max(axis=0)
                
                # 1. Save true physical bounds
                save_path = Path(__file__).resolve().parent.parent / "Data" / "state_norm_params.json"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "w") as f:
                    json.dump({"mins": mins.tolist(), "maxs": maxs.tolist()}, f, indent=2)
                
                # 2. Apply Normalization to DataFrame IN-PLACE
                denom = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
                dataframe[STATE_COLS] = ((dataframe[STATE_COLS] - mins) / denom).astype(np.float32)
                if all(c in dataframe.columns for c in NEXT_STATE_COLS):
                    dataframe[NEXT_STATE_COLS] = ((dataframe[NEXT_STATE_COLS] - mins) / denom).astype(np.float32)
                
                print(f"[+] FINAL Normalization Sweep complete. True Physical Max Queue saved: {maxs[8]:.1f}")
                
            # Now explicitly save the cleanly normalized frame
            if args.output is not None:
                from utils.offline_rl.io import save_transition_dataframe
                save_transition_dataframe(
                    dataframe, 
                    args.output, 
                    config_fingerprint=config_hash, 
                    compute_reward_stats=True
                )
        except ValueError as exc:
            print(f"Transition build configuration/data error: {exc}")
            return 1

        print(
            "Built transition dataset from raw parquet "
            f"with {len(dataframe)} rows."
        )
    else:
        if not args.input.exists():
            print(f"Input parquet not found: {args.input}")
            return 1
        dataframe = load_transition_dataframe(args.input)
        
        # === INVERSE NORMALIZATION (Phục hồi vật lý) ===
        # Do file Parquet v28 đã bị chuẩn hóa sẵn về 0-1, nhưng Scaler của d3rlpy 
        # lại dùng mins/maxs vật lý thô, nên chúng ta cần nhân ngược lại để AI "sáng mắt".
        try:
            from utils.offline_rl.schema import STATE_COLS
            from pathlib import Path
            
            BASE_DIR = Path(__file__).resolve().parent
            # Gọi trực tiếp hàm nội bộ đã được định nghĩa ở dòng 251
            mins_phys, maxs_phys = load_normalization_params(BASE_DIR.parent / "Data")
            
            print(f"[*] Đang thực hiện Inverse Normalization cho {len(STATE_COLS)} cột tính năng...")
            for i, col in enumerate(STATE_COLS):
                if col in dataframe.columns:
                    # Công thức Inverse: X_raw = X_norm * (Max - Min) + Min
                    # Vì trong build_state_action chỉ chia cho Max nên ta chỉ cần nhân Max 
                    # (Hoặc chuẩn nhất là dùng đúng công thức d3rlpy đang kỳ vọng)
                    dataframe[col] = dataframe[col] * (maxs_phys[i] - mins_phys[i]) + mins_phys[i]
            
            print("✅ Phục hồi vật lý THÀNH CÔNG. AI đã có thể nhìn thấy dữ liệu thô.")
        except Exception as e:
            print(f"[!] Cảnh báo Inverse Normalization thất bại: {e}. AI có thể vẫn bị 'mù'.")

    if args.skip_validation:
        print("[!] Skipping data validation as requested. Startup will be faster.")
    else:
        result = validate_transition_dataframe(dataframe, mode=args.mode)
        print(format_validation_report(result))

        if not result.passed:
            print("Validation failed. Training is blocked until required transition fields are fixed.")
            return 2

    if args.evaluate:
        return run_evaluation(dataframe, args)

    # Toy training placeholder - this is where you would call your training function after validation passes
    if args.train_toy:
        print("=== Toy Training Config ===")
        print(f"Sample Size: {args.sample_size}")
        print(f"Number of Steps: {args.n_steps}")
        print(f"Seed: {args.seed}")
        print(f"Validation Mode: {args.mode}")
        print("===========================")

        return train_toy_iql(dataframe, args)
    
    if not args.to_d3rlpy:
        print("Validation passed. Skip d3rlpy conversion because --to-d3rlpy was not requested.")
        return 0

    try:
        dataset = build_d3rlpy_dataset(dataframe, mode=args.mode)
    except TransitionValidationError as exc:
        print(f"d3rlpy conversion blocked by validation errors: {exc}")
        return 3
    except RuntimeError as exc:
        print(f"d3rlpy runtime dependency error: {exc}")
        return 3
    except ValueError as exc:
        print(f"d3rlpy conversion value error: {exc}")
        return 3

    print(
        "d3rlpy dataset created successfully "
        f"with {len(dataframe)} transitions and type {type(dataset).__name__}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
