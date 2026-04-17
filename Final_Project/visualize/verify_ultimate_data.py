import pandas as pd
import json
import numpy as np
from pathlib import Path

def verify_dataset(parquet_path, json_path):
    print("="*50)
    print("🕵️  BỘ KIỂM ĐỊNH DATA TỐI THƯỢNG (ULTIMATE VERIFIER)")
    print("="*50)

    try:
        df = pd.read_parquet(parquet_path)
        with open(json_path, "r") as f:
            norm_params = json.load(f)
    except Exception as e:
        print(f"❌ Lỗi tải file: {e}")
        return False

    passed_all = True

    # 1. Kiểm tra sự tồn tại của Nhãn Oracle
    print("\n[1] KIỂM TRA NHÃN ORACLE (POLICY_TYPE)")
    if "policy_type" not in df.columns:
        print("❌ FAILED: Không có cột policy_type. Dữ liệu chưa qua Oracle!")
        passed_all = False
    else:
        counts = df["policy_type"].value_counts().to_dict()
        print(f"✅ PASSED: Cột policy_type tồn tại. Phân bổ: {counts}")

    # 2. Kiểm tra Normalization Integrity (Dải chuẩn hóa Max Queue)
    print("\n[2] KIỂM TRA BẢO TOÀN CHUẨN HÓA VẬT LÝ")
    Q_IDX = 8
    max_queue_norm = norm_params["maxs"][Q_IDX]
    if max_queue_norm < 1000:
        print(f"❌ FAILED: Max Queue chuẩn hóa rác: {max_queue_norm} (Quá thấp, đã bị nhiễm Oracle bias!)")
        passed_all = False
    else:
        print(f"✅ PASSED: Bộ chia chuẩn hóa (Max Queue) giữ được vật lý gốc: {max_queue_norm:,.1f}")

    # 3. Kiểm tra rò rỉ dữ liệu (Data Leakage) ở cột s_queue
    print("\n[3] KIỂM TRA RÒ RỈ CHUẨN HÓA (s_queue > 1.0)")
    s_queue_max = df["s_queue"].max()
    s_queue_min = df["s_queue"].min()
    if s_queue_max > 1.01 or s_queue_min < -0.01:
        print(f"❌ FAILED: s_queue bị tràn khung giới hạn! Min={s_queue_min:.4f}, Max={s_queue_max:.4f}")
        passed_all = False
    else:
        print(f"✅ PASSED: s_queue hoàn toàn nằm gọn trong dải [0, 1]. Max thực tế = {s_queue_max:.4f}")

    # 4. Kiểm định chéo Toán học (Arrival Scale = 0.05)
    print("\n[4] KIỂM TOÁN LẠI TOÁN HỌC (ARRIVAL SCALE XÁC MINH)")
    # Lấy 1 episode bất kỳ để test
    ep_df = df[df["episode_id"] == df["episode_id"].unique()[0]].reset_index(drop=True)
    
    # Tính lại queue size ảo
    test_passed = True
    for i in range(1, 10): # Test 10 step đầu
        tx_count = ep_df.loc[i, "transaction_count"]
        arrival_physical = tx_count * 0.05
        
        # This is a soft check, because actual execution limits and rounding might drift slowly.
        # But we verify that arrival scale is indeed operating in the ~5% magnitude.
        queue_increase_if_no_exec = ep_df.loc[i-1, "queue_size"] + arrival_physical
        
        if queue_increase_if_no_exec < ep_df.loc[i, "queue_size"]:
            print(f"❌ FAILED: Ở step {i}, hàng đợi thực tế ({ep_df.loc[i, 'queue_size']}) CÒN LỚN HƠN tổng dồn. Dấu hiệu dùng scale sai (VD 0.5)!")
            test_passed = False
            passed_all = False
            break
            
    if test_passed:
        print(f"✅ PASSED: Toán học hàng đợi khớp với arrival_scale=0.05.")

    print("\n" + "="*50)
    if passed_all:
        print("🏆 KẾT LUẬN: BỘ DỮ LIỆU ĐẠT CHUẨN VÀNG (GOLDEN STANDARD). SẴN SÀNG TRAINING!")
    else:
        print("☠️ KẾT LUẬN: BỘ DỮ LIỆU VẪN LỖI. VUI LÒNG KIỂM TRA LẠI CODE.")
        
    return passed_all

if __name__ == "__main__":
    import sys
    parquet_file = "Final_Project/Data/transitions_discrete_v21_TRUE_ULTIMATE.parquet"
    json_file = "Final_Project/Data/state_norm_params.json"
    verify_dataset(parquet_file, json_file)
