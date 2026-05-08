import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

# Đảm bảo import được các module
# PROJECT_ROOT sẽ là thư mục 'labs'
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import các hàm cốt lõi (SOTA V28 logic)
from Final_Project.code.utils.load_data import TransitionBuildConfig, build_transitions

def build_v29_pipeline():
    print("🚀 Bắt đầu quy trình xây dựng dữ liệu V29 (Raw Mathematical)...")
    
    # 1. Cấu hình (Tắt normalize_state để lấy số thô)
    config = TransitionBuildConfig(normalize_state=False)
    
    # Đường dẫn (Tự động tìm file raw trong Final_Project/Data)
    data_dir = PROJECT_ROOT / "Final_Project" / "Data"
    raw_data_path = data_dir / "data_2024-04-10_2026-04-10.parquet"
    output_parquet = data_dir / "data_v29_raw.parquet"
    output_json = data_dir / "state_norm_params.json" # Ghi đè để các script khác tự nhận

    if not raw_data_path.exists():
        print(f"❌ Không tìm thấy file dữ liệu thô tại: {raw_data_path}")
        return

    # 2. Đọc dữ liệu
    print(f"[1/4] Đang đọc dữ liệu thô: {raw_data_path.name}")
    raw_df = pd.read_parquet(raw_data_path)

    # 3. Build Transitions
    print("[2/4] Đang xử lý logic V28 (Log, Momentum, Surprise, Backlog)...")
    v29_df = build_transitions(
        raw_df, 
        config, 
        use_oracle=True, # Dùng Oracle để có dữ liệu Expert cho Offline RL
        expert_ratio=0.5,
        medium_ratio=0.3,
        random_ratio=0.2
    )

    # 4. Tính toán thông số chuẩn hóa mới
    print("[3/4] Đang trích xuất Min/Max cho bộ Scaler mới...")
    from Final_Project.code.utils.offline_rl.schema import STATE_COLS
    state_data = v29_df[STATE_COLS].to_numpy(dtype=np.float32)
    mins = state_data.min(axis=0).tolist()
    maxs = state_data.max(axis=0).tolist()

    # 5. Xuất bản
    print(f"[4/4] Đang lưu trữ kết quả...")
    
    # Lưu JSON
    with open(output_json, "w") as f:
        json.dump({"mins": mins, "maxs": maxs}, f, indent=4)
    
    # Lưu Parquet
    v29_df.to_parquet(output_parquet)
    
    print("\n" + "⭐" * 30)
    print("✅ HOÀN THÀNH V29!")
    print(f"   - Parquet: {output_parquet.name}")
    print(f"   - Scaler JSON: {output_json.name}")
    print(f"   - Số lượng transitions: {len(v29_df)}")
    print("⭐" * 30)

if __name__ == "__main__":
    build_v29_pipeline()
