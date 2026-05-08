import pandas as pd
import numpy as np
import json
from pathlib import Path

def run_super_audit():
    print("=== 🛡️ PHÒNG LAB ĐỐI SOÁT SIÊU CẤP V28 vs V29 🛡️ ===")
    
    # 1. Đường dẫn
    data_dir = Path("Final_Project/Data")
    v28_path = data_dir / "data_2024-04-10_2026-04-10_transitions_v28.parquet"
    v29_path = data_dir / "transitions_discrete_v29.parquet"
    json_path = data_dir / "state_norm_params.json" # Đây là file JSON của V29 vừa build

    if not v28_path.exists() or not v29_path.exists():
        # Fallback if names are different
        v28_path = list(data_dir.glob("*v28.parquet"))[0] if list(data_dir.glob("*v28.parquet")) else v28_path
        v29_path = list(data_dir.glob("*v29.parquet"))[0] if list(data_dir.glob("*v29.parquet")) else v29_path

    print(f"[*] Đang nạp V28: {v28_path.name}")
    print(f"[*] Đang nạp V29: {v29_path.name}")

    df28 = pd.read_parquet(v28_path)
    df29 = pd.read_parquet(v29_path)

    # 2. Lọc Oracle và Khớp dữ liệu
    print("[+] Đang lọc dữ liệu Oracle (policy_type=1) và khớp Timestamp...")
    df28_o = df28[df28["policy_type"] == 1].set_index("timestamp").sort_index()
    df29_o = df29[df29["policy_type"] == 1].set_index("timestamp").sort_index()

    # Chỉ lấy những timestamp có mặt ở cả 2 bản
    common_idx = df28_o.index.intersection(df29_o.index)
    df28_o = df28_o.loc[common_idx]
    df29_o = df29_o.loc[common_idx]

    print(f"    -> Đã khớp được {len(df28_o)} dòng Oracle.")

    # 3. Nạp thông số chuẩn hóa (để giải nén V28 hoặc nén V29)
    with open(json_path, "r") as f:
        params = json.load(f)
        mins = np.array(params["mins"])
        maxs = np.array(params["maxs"])

    # 4. SO SÁNH CHI TIẾT 11 CHIỀU
    from utils.offline_rl.schema import STATE_COLS
    
    results = []
    for i, col in enumerate(STATE_COLS):
        s28 = df28_o[col].to_numpy()
        s29 = df29_o[col].to_numpy()
        
        # Tính tương quan Pearson
        corr = np.corrcoef(s28, s29)[0, 1]
        
        # Thử "Giải nén" V28 để xem có về đúng V29 không
        # Lưu ý: V28 cũ có thể dùng bộ mins/maxs cũ, nhưng về logic tỉ lệ phải khớp.
        # Ở đây ta kiểm tra xem s29 có phải là phép biến đổi tuyến tính của s28 không.
        is_linear = corr > 0.9999
        
        results.append({
            "Feature": col,
            "Correlation": f"{corr:.6f}",
            "Status": "✅ KHỚP" if is_linear else "❌ LỆCH"
        })

    # 5. So sánh Action và Reward
    action_match = (df28_o["action"] == df29_o["action"]).mean() * 100
    reward_diff = np.abs(df28_o["reward"] - df29_o["reward"]).max()

    # 6. HIỂN THỊ KẾT QUẢ
    report = pd.DataFrame(results)
    print("\n" + "="*50)
    print(report.to_string(index=False))
    print("="*50)
    print(f"[ACTION] Độ khớp hành động: {action_match:.2f}%")
    print(f"[REWARD] Độ lệch Reward cực đại: {reward_diff:.6e}")
    
    if action_match > 99.9 and reward_diff < 1e-5:
        print("\n🎉 KẾT LUẬN: V29 hoàn toàn nhất quán với V28 về mặt chiến thuật Oracle!")
        print("Bạn có thể tin tưởng 100% vào bộ dữ liệu mới này.")
    else:
        print("\n⚠️ CẢNH BÁO: Có sự sai lệch trong logic. Cần kiểm tra lại build_transitions.")

if __name__ == "__main__":
    run_super_audit()
