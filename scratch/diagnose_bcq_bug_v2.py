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

def diagnose_v2():
    model_path = "good_models/DiscreteBCQ_V6_20260418_1044_20260418104436/model_10000.d3"
    data_path = "Final_Project/Data/transitions_discrete_v28.parquet"
    norm_path = "Final_Project/Data/state_norm_params.json"

    print("--- 🔍 CHẨN ĐOÁN LỖI LEADERBOARD BCQ (V2) ---")
    
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
    
    # Ép buộc normalize_state = True để so sánh tác động của mins/maxs
    base_config = TransitionBuildConfig()
    config = dataclasses.replace(base_config, normalize_state=True)

    # === THÍ NGHIỆM 1: Lỗi Mins=None (Leaderboard hiện tại) ===
    print("\n[THÍ NGHIỆM 1] Mins=None (Logic Leaderboard hiện tại):")
    env_fail = CharityGasEnv(ep_df, config, mins=None, maxs=None)
    obs, _ = env_fail.reset()
    
    for i in range(3):
        action = algo.predict(obs.reshape(1, -1))[0]
        q_val = obs[8] 
        print(f"  > Bước {i}: AI nhận Queue = {q_val:10.4f} | AI chọn Action = {action}")
        obs, _, _, _, _ = env_fail.step(action)

    # === THÍ NGHIỆM 2: Có Mins/Maxs (Cấu hình chuẩn) ===
    print("\n[THÍ NGHIỆM 2] Có Mins/Maxs (Dữ liệu đã chuẩn hóa chuẩn):")
    env_ok = CharityGasEnv(ep_df, config, mins=mins, maxs=maxs)
    obs, _ = env_ok.reset()
    
    for i in range(3):
        action = algo.predict(obs.reshape(1, -1))[0]
        q_val = obs[8]
        print(f"  > Bước {i}: AI nhận Queue = {q_val:10.6f} | AI chọn Action = {action}")
        obs, _, _, _, _ = env_ok.step(action)

    print("\n--- 🏁 PHÂN TÍCH ---")
    print("Bác xem: Ở Bước 1 của TN1, AI nhận số 16.0 (Vật lý) -> Nó bị 'sốc' và hành động sai.")
    print("Ở Bước 1 của TN2, AI nhận số 0.000686 (Chuẩn hóa) -> Đây mới là thang đo nó được học!")

if __name__ == "__main__":
    diagnose_v2()
