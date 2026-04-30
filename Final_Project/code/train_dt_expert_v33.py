import os
import sys
import pandas as pd
import numpy as np
import d3rlpy
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteDecisionTransformer
from d3rlpy.optimizers import AdamWConfig
import torch

# Đảm bảo hệ thống dùng đúng số luồng yêu cầu
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
torch.set_num_threads(2)

def train_dt():
    input_path = "Final_Project/Data/transitions_v33_L2_Batching_RAW.parquet"
    print(f"[*] Đang nạp dữ liệu từ {input_path}...")
    df = pd.read_parquet(input_path)

    # 1. CHIA TÁCH TRAIN/TEST ĐỂ TRÁNH LEAKAGE
    unique_eps = sorted(df['episode_id'].unique())
    train_ids = unique_eps[:int(len(unique_eps) * 0.8)] # Chỉ lấy 80% đầu để train
    
    train_df = df[df['episode_id'].isin(train_ids)].copy()
    print(f"[*] Tổng số Episode khả dụng để train: {len(train_ids)}")

    # 2. CHIẾN THUẬP: GẠN ĐỤC KHƠI TRONG (Chỉ trên Train Set)
    print("[*] Đang lọc 40% dữ liệu tinh hoa (Expert) từ tập Train...")
    ep_rewards = train_df.groupby('episode_id')['reward'].sum()
    threshold = ep_rewards.quantile(0.6) 
    expert_episodes = ep_rewards[ep_rewards >= threshold].index
    
    expert_df = train_df[train_df['episode_id'].isin(expert_episodes)].copy()
    print(f"✅ Đã lọc xong: {len(expert_episodes)} episodes Expert ({len(expert_df):,} transitions)")

    # 2. CHUYỂN ĐỔI SANG MDPDATASET (Dạng mà d3rlpy DT yêu cầu)
    # DT cần dữ liệu dạng chuỗi (Sequence)
    observations = expert_df[['s_queue', 's_base_fee', 's_gas_limit', 's_urgency', 's_gas_ma', 's_gas_std']].values
    actions = expert_df['action'].values
    rewards = expert_df['reward'].values
    terminals = expert_df['terminal'].values
    episode_terminals = expert_df['terminal'].values # Hoặc tính toán dựa trên episode_id change

    dataset = MDPDataset(
        observations=observations.astype(np.float32),
        actions=actions.astype(np.int32),
        rewards=rewards.astype(np.float32),
        terminals=terminals.astype(np.float32),
    )

    # 3. CẤU HÌNH DECISION TRANSFORMER
    print("[*] Khởi tạo Decision Transformer...")
    dt = DiscreteDecisionTransformer(
        batch_size=64,
        learning_rate=1e-4,
        optim_config=AdamWConfig(weight_decay=1e-4),
        context_size=20,       # Nhìn lại 20 bước lịch sử để đoán tương lai
        num_heads=4,
        num_layers=3,
        embed_dim=128,
        observation_scaler="robust",
    )

    # 4. TRAINING
    print("[*] Bắt đầu huấn luyện Decision Transformer (Expert Only)...")
    # Chúng ta đặt Target Return cao (ví dụ 1000) để ép nó học theo Oracle
    dt.fit_transformer(
        dataset,
        n_steps=300000,        # DT hội tụ khá nhanh trên dữ liệu sạch
        n_steps_per_epoch=1000,
        save_interval=10,      # Lưu mỗi 10 epoch
    )

    print("✅ HOÀN THÀNH TRAINING DECISION TRANSFORMER!")

if __name__ == "__main__":
    train_dt()
