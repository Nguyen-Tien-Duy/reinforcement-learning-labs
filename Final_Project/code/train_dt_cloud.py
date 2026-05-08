import argparse
import logging
from pathlib import Path
from datetime import datetime
import gc

import numpy as np
import pandas as pd
import d3rlpy
import torch
import json
from d3rlpy.algos import DiscreteDecisionTransformerConfig
from d3rlpy.preprocessing import MinMaxObservationScaler, StandardRewardScaler

# Cấu hình log
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

STATE_COLS = [
    "s_gas_t0", "s_gas_t1", "s_gas_t2",       
    "s_congestion", "s_momentum", "s_accel",   
    "s_surprise", "s_backlog",               
    "s_queue", "s_time_left", "s_gas_ref"
]

def build_d3rlpy_dataset(df: pd.DataFrame):
    from d3rlpy import ActionSpace
    from d3rlpy.dataset import MDPDataset

    observations = df[STATE_COLS].to_numpy(dtype=np.float32)
    rewards = pd.to_numeric(df["reward"], errors="coerce").to_numpy(dtype=np.float32)
    actions = df["action"].to_numpy(dtype=np.int64)
    terminals = df["done"].to_numpy(dtype=np.float32)

    return MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        action_space=ActionSpace.DISCRETE
    )

def load_normalization_params(data_dir: Path):
    params_path = data_dir / "state_norm_params.json"
    with open(params_path, "r") as f:
        params = json.load(f)
    return np.array(params["mins"], dtype=np.float32), np.array(params["maxs"], dtype=np.float32)

def main():
    parser = argparse.ArgumentParser(description="Train Discrete Decision Transformer on Cloud H100")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("dt_output"))
    parser.add_argument("--n-steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--context-size", type=int, default=50)
    parser.add_argument("--oracle-only", action="store_true")
    args = parser.parse_args()

    logging.info(f"[*] Nạp dữ liệu từ {args.input}")
    df = pd.read_parquet(args.input)

    # === 1. CHIA TÁCH TRAIN/TEST (CHẶN LEAKAGE) ===
    unique_episodes = sorted(df["episode_id"].unique())
    total_episodes = len(unique_episodes)
    split_idx = int(total_episodes * 0.8)
    
    train_pool_ids = unique_episodes[:split_idx]
    test_pool_ids = unique_episodes[split_idx:]
    
    logging.info("="*50)
    logging.info(f"📊 BÁO CÁO PHÂN LẬP DỮ LIỆU (CHỐNG LEAKAGE)")
    logging.info(f"  > Tổng số Episodes trong file RAW: {total_episodes:,}")
    logging.info(f"  > Số Episodes tập TEST (bị khóa/cất đi): {len(test_pool_ids):,} (Chiếm 20% cuối)")
    logging.info(f"  > Số Episodes tập TRAIN (được dùng để lọc): {len(train_pool_ids):,} (Chiếm 80% đầu)")
    logging.info("="*50)
    
    train_pool_df = df[df["episode_id"].isin(train_pool_ids)].copy()
    del df
    gc.collect()

    # === 2. LỌC EXPERT ===
    if args.oracle_only:
        logging.info("[*] Đang lọc EXPERT (Policy Type == 1) CHỈ TỪ TẬP TRAIN...")
        train_pool_df = train_pool_df[train_pool_df["policy_type"] == 1].copy()
        expert_eps = train_pool_df["episode_id"].nunique()
        logging.info(f"✅ BÁO CÁO SAU KHI LỌC EXPERT:")
        logging.info(f"  > Số Episodes Expert giữ lại: {expert_eps:,} / {len(train_pool_ids):,} (Train Pool)")
        logging.info(f"  > Số lượng Transitions (Dòng): {len(train_pool_df):,} transitions.")
        logging.info("="*50)

    # === 3. XỬ LÝ REWARD SCALING CHO TRANSFORMER ===
    logging.info("[*] Khởi tạo Reward Scaler (Chống nổ Attention)...")
    raw_rewards = train_pool_df["reward"].to_numpy()
    # Log-transform như cũ để giảm bớt độ lệch
    log_rewards = np.sign(raw_rewards) * np.log1p(np.abs(raw_rewards))
    train_pool_df["reward"] = log_rewards
    
    median = float(np.median(log_rewards))
    q75, q25 = np.percentile(log_rewards, [75, 25])
    iqr = float(q75 - q25) if float(q75 - q25) > 0 else 1.0
    rew_scaler = StandardRewardScaler(mean=median, std=iqr)

    # === 4. BUILD DATASET ===
    train_dataset = build_d3rlpy_dataset(train_pool_df)
    del train_pool_df
    gc.collect()

    # === 5. CẤU HÌNH DT CHO H100 ===
    logging.info("[*] Nạp State Normalization Parameters...")
    mins, maxs = load_normalization_params(args.input.parent)
    obs_scaler = MinMaxObservationScaler(minimum=mins, maximum=maxs) 
    
    config = DiscreteDecisionTransformerConfig(
        batch_size=args.batch_size,
        learning_rate=1e-4,
        context_size=args.context_size,
        max_timestep=10000,
        num_heads=8,
        num_layers=10,
        observation_scaler=obs_scaler,
        reward_scaler=rew_scaler,
        compile_graph=True,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info(f"[*] Training on Device: {device}")
    
    algo = config.create(device=device)
    algo.build_with_dataset(train_dataset)

    args.outdir.mkdir(parents=True, exist_ok=True)
    run_name = f"DT_V33_H100_{datetime.now().strftime('%m%d_%H%M')}"
    
    logging.info(f"🚀 BẮT ĐẦU HUẤN LUYỆN ({args.n_steps} steps)...")
    algo.fit(
        train_dataset,
        n_steps=args.n_steps,
        n_steps_per_epoch=5000,
        show_progress=True,
        experiment_name=run_name
    )
    
    logging.info("✅ HOÀN THÀNH!")

if __name__ == "__main__":
    main()
