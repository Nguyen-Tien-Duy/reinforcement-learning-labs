import pandas as pd
import numpy as np
import d3rlpy
import torch
import json
import dataclasses
from pathlib import Path
import sys
import os

# Thêm đường dẫn để import được utils
sys.path.append(os.path.abspath("Final_Project/code"))

from utils.offline_rl.enviroment import CharityGasEnv
from utils.offline_rl.config import TransitionBuildConfig

def diagnose_v3():
    model_path = "good_models/DiscreteBCQ_V6_20260418_1044_20260418104436/model_10000.d3"
    data_path = "Final_Project/Data/transitions_discrete_v28.parquet"
    norm_path = "Final_Project/Data/state_norm_params.json"

    print("--- 🔍 CHẨN ĐOÁN LỖI LEADERBOARD BCQ (V3) ---")
    
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
    
    base_config = TransitionBuildConfig()

    # === THÍ NGHIỆM 1: MÔ PHỎNG CHÍNH XÁC LỖI CỦA LEADERBOARD HIỆN TẠI ===
    print("\n[THÍ NGHIỆM 1] Mô phỏng lỗi của Leaderboard hiện tại (normalize_state=False):")
    config_fail = dataclasses.replace(base_config, normalize_state=False)
    env_fail = CharityGasEnv(ep_df, config_fail, mins=None, maxs=None)
    obs, _ = env_fail.reset()
    
    for i in range(2):
        action = algo.predict(obs.reshape(1, -1))[0]
        q_val = obs[8] 
        print(f"  > Bước {i}: AI nhận Queue = {q_val:10.4f} | AI chọn Action = {action}")
        obs, _, _, _, _ = env_fail.step(action)

    # === THÍ NGHIỆM 2: CHẠY ĐÚNG THIẾT KẾ (normalize_state=True + mins/maxs) ===
    print("\n[THÍ NGHIỆM 2] Chạy ĐÚNG thiết kế (normalize_state=True + mins/maxs):")
    config_ok = dataclasses.replace(base_config, normalize_state=True)
    env_ok = CharityGasEnv(ep_df, config_ok, mins=mins, maxs=maxs)
    obs, _ = env_ok.reset()
    
    for i in range(2):
        action = algo.predict(obs.reshape(1, -1))[0]
        q_val = obs[8]
        # In thêm cả giá trị queue thực tế bên trong Env để bác thấy
        real_q = env_ok.queue_size
        print(f"  > Bước {i}: AI nhận Queue = {q_val:10.6f} (Vật lý thật: {real_q}) | AI chọn Action = {action}")
        obs, _, _, _, _ = env_ok.step(action)

    print("\n--- 🏁 PHÁN QUYẾT TỪ CON SỐ ---")
    print("Bác thấy TN1 không? Bước 0 nhận số nhỏ (0.0011), Bước 1 nhận số to (16.0000). AI bị 'đánh lừa'!")
    print("Ở TN2, AI luôn nhận được số đã nén chuẩn (0.000047 và 0.000686). Đây mới là sự thật!")

if __name__ == "__main__":
    diagnose_v3()
