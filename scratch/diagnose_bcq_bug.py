import pandas as pd
import numpy as np
import d3rlpy
import torch
import json
from pathlib import Path
import sys
import os

# Thêm đường dẫn để import được utils
sys.path.append(os.path.abspath("Final_Project/code"))

from utils.offline_rl.enviroment import CharityGasEnv
from utils.offline_rl.config import TransitionBuildConfig

def diagnose():
    model_path = "good_models/DiscreteBCQ_V6_20260418_1044_20260418104436/model_10000.d3"
    data_path = "Final_Project/Data/transitions_discrete_v28.parquet"
    norm_path = "Final_Project/Data/state_norm_params.json"

    print("--- 🔍 CHẨN ĐOÁN LỖI LEADERBOARD BCQ ---")
    
    # 1. Load Model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    algo = d3rlpy.load_learnable(model_path, device=device)
    
    # 2. Load Data & Params
    df = pd.read_parquet(data_path)
    with open(norm_path, "r") as f:
        params = json.load(f)
    mins = np.array(params["mins"], dtype=np.float32)
    maxs = np.array(params["maxs"], dtype=np.float32)

    ep_id = df["episode_id"].unique()[0]
    ep_df = df[df["episode_id"] == ep_id].copy().reset_index(drop=True)
    config = TransitionBuildConfig()
    import dataclasses
    config = dataclasses.replace(config, normalize_state=False)

    # === THÍ NGHIỆM 1: Cấu hình LỖI (Mins=None) ===
    print("\n[THÍ NGHIỆM 1] Chạy với cấu hình HIỆN TẠI (Lỗi Mins=None):")
    env_fail = CharityGasEnv(ep_df, config, mins=None, maxs=None)
    obs, _ = env_fail.reset()
    
    for i in range(3):
        # AI dự đoán dựa trên observation nhận được
        action = algo.predict(obs.reshape(1, -1))[0]
        
        # Lấy giá trị Hàng đợi s_queue (Index 8)
        q_val = obs[8] 
        print(f"  > Bước {i}: Queue nhận được = {q_val:10.4f} | AI chọn Action = {action}")
        
        obs, _, _, _, _ = env_fail.step(action)

    # === THÍ NGHIỆM 2: Cấu hình ĐÚNG (Có Mins/Maxs) ===
    print("\n[THÍ NGHIỆM 2] Chạy với cấu hình ĐÃ SỬA (Có Mins/Maxs):")
    env_ok = CharityGasEnv(ep_df, config, mins=mins, maxs=maxs)
    obs, _ = env_ok.reset()
    
    for i in range(3):
        action = algo.predict(obs.reshape(1, -1))[0]
        q_val = obs[8]
        print(f"  > Bước {i}: Queue nhận được = {q_val:10.4f} | AI chọn Action = {action}")
        
        obs, _, _, _, _ = env_ok.step(action)

    print("\n--- 🏁 KẾT LUẬN ---")
    print("Bác thấy sự khác biệt ở Bước 1 chưa? Thí nghiệm 1 bộc lộ lỗ hổng quy chiếu cực lớn!")

if __name__ == "__main__":
    diagnose()
