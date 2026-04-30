import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import d3rlpy

# Đảm bảo import được các module từ Final_Project/code
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if 'Final_Project/code' not in sys.path:
    sys.path.append(os.path.join(PROJECT_ROOT, 'Final_Project', 'code'))

from utils.offline_rl.enviroment import CharityGasEnv
from utils.offline_rl.config import TransitionBuildConfig

# ======================================================================
# 🔒 ĐÓNG BĂNG TÀI NGUYÊN (FROZEN ASSETS)
# Không thay đổi các đường dẫn này để đảm bảo tính tái lập 100%
# ======================================================================
# 1. Dữ liệu đánh giá (Tập Test 20% cuối cùng - 242 episodes)
DATA_PATH = os.path.join(PROJECT_ROOT, 'Final_Project', 'Data', 'transitions_v33_L2_Batching_RAW.parquet')

# 2. Model Chiến Thắng (SOTA - Huấn luyện với Reward Shaping đúng đắn)
MODEL_PATH = os.path.join(PROJECT_ROOT, 'd3rlpy_logs', 'DiscreteCQL_V6_20260428_0426_20260428042630', 'model_160000.d3')

# 3. Parameters chuẩn hóa trạng thái (Normalization params)
NORM_PATH = os.path.join(PROJECT_ROOT, 'Final_Project', 'Data', 'state_norm_params.json')

# Cấu hình Safety Layer (Ép bán tháo khi còn <= 20% thời gian)
SAFETY_THRESHOLD = 0.20
# ======================================================================

def main():
    print(f"[{'*'*60}]")
    print("🚀 SCRIPT TÁI LẬP KẾT QUẢ SOTA (STATE-OF-THE-ART)")
    print("Chiến lược: CQL model_160000 + Safety Layer 20%")
    print(f"[{'*'*60}]\n")

    # 1. Kiểm tra sự tồn tại của file
    for path, name in [(DATA_PATH, "Data"), (MODEL_PATH, "Model"), (NORM_PATH, "Norm Params")]:
        if not os.path.exists(path):
            print(f"❌ LỖI: Không tìm thấy file {name} tại {path}")
            return
    print("✅ Đã tìm thấy tất cả các file đóng băng (Data, Model, Norm Config).")

    # 2. Load dữ liệu
    print("⏳ Đang load dữ liệu...")
    df = pd.read_parquet(DATA_PATH)
    unique_eps = sorted(df['episode_id'].unique())
    # Lấy 20% episodes cuối cùng làm tập test (hold-out)
    test_ids = unique_eps[int(len(unique_eps) * 0.8):]
    ep_list = [d.reset_index(drop=True) for _, d in df[df['episode_id'].isin(test_ids)].groupby('episode_id')]
    
    print(f"✅ Đã load {len(ep_list)} episodes Test (Từ ID {test_ids[0]} đến {test_ids[-1]}).")

    # 3. Load cấu hình và Model
    print("⏳ Đang khởi tạo môi trường và Model...")
    config = TransitionBuildConfig()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = d3rlpy.load_learnable(MODEL_PATH, device=device)
    print(f"✅ Model load thành công trên thiết bị: {device}")

    # 4. Chạy mô phỏng đánh giá
    print("\n🏃 Đang mô phỏng các chiến lược trên toàn bộ tập Test. Vui lòng đợi...\n")
    
    greedy_costs = []
    cql_costs = []
    cql_misses = []
    
    for i, ep_df in enumerate(ep_list):
        if (i + 1) % 50 == 0:
            print(f"   Đã xử lý {i + 1}/{len(ep_list)} episodes...")

        # --- A. Đánh giá Greedy (Baseline) ---
        env = CharityGasEnv(ep_df, config, mins=None, maxs=None)
        env.reset()
        done = False
        gc = 0.0
        while not done:
            # Action 4: Xả 100% hàng ngay lập tức
            _, _, ter, tru, info = env.step(4)
            gc += info.get('cost', 0.0)
            if ter or tru:
                done = True
        greedy_costs.append(gc)
        
        # --- B. Đánh giá CQL + Safety Layer (SOTA) ---
        env = CharityGasEnv(ep_df, config, mins=None, maxs=None)
        obs, _ = env.reset()
        done = False
        cc = 0.0
        miss = False
        
        while not done:
            time_ratio = env.time_to_deadline / env.config.episode_hours
            
            # Kích hoạt Safety Layer
            if time_ratio < SAFETY_THRESHOLD:
                action = 4 # Ép xả 100%
            else:
                # CQL Model dự đoán
                obs_c = np.ascontiguousarray(obs.reshape(1, -1), dtype=np.float32)
                action = int(model.predict(obs_c)[0])
                
            obs, _, ter, tru, info = env.step(action)
            cc += info.get('cost', 0.0)
            
            if ter or tru:
                miss = info.get('deadline_miss', False)
                done = True
                
        cql_costs.append(cc)
        cql_misses.append(1 if miss else 0)

    # 5. Tổng hợp kết quả
    mean_greedy = np.mean(greedy_costs)
    mean_cql = np.mean(cql_costs)
    miss_rate = np.mean(cql_misses) * 100
    savings_pct = ((mean_greedy - mean_cql) / mean_greedy) * 100

    print(f"\n[{'='*60}]")
    print(f"🏆 KẾT QUẢ CUỐI CÙNG TRÊN TẬP TEST ĐỘC LẬP ({len(ep_list)} episodes)")
    print(f"[{'='*60}]")
    print(f"1. Greedy Baseline:    {mean_greedy:>10,.0f} Gwei")
    print(f"2. CQL + Safety 20%:   {mean_cql:>10,.0f} Gwei")
    print(f"3. Deadline Miss Rate: {miss_rate:>10.1f} %")
    print("-" * 62)
    
    if mean_cql < mean_greedy:
        print(f"⭐ HIỆU NĂNG: CQL tiết kiệm được +{savings_pct:.1f}% chi phí Gas so với Greedy!")
    else:
        print(f"❌ HIỆU NĂNG: CQL đắt hơn Greedy {-savings_pct:.1f}%.")
    
    print(f"[{'='*60}]\n")

if __name__ == "__main__":
    main()
