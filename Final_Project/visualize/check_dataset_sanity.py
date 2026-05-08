import pandas as pd
import numpy as np
import json
from pathlib import Path

def check_sanity(parquet_path):
    print(f"\n[*] Đang kiểm tra độ 'sạch' của dữ liệu: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    STATE_COLS = ["s_gas_t0", "s_gas_t1", "s_gas_t2", "s_congestion", "s_momentum", 
                  "s_accel", "s_surprise", "s_backlog", "s_queue", "s_time_left", "s_gas_ref"]
    
    # 1. Kiểm tra Shape
    print(f"[1] Tổng số dòng: {len(df):,}")
    
    # 2. Kiểm tra các giá trị Normalized (Phải nằm trong 0..1 hoặc xấp xỉ)
    print("\n[2] Kiểm tra dải giá trị Normalized State (Kỳ vọng 0.0 - 1.0):")
    for col in STATE_COLS:
        if col in df.columns:
            c_min = df[col].min()
            c_max = df[col].max()
            status = "✓ OK" if (0 <= c_min <= 1.05 and 0 <= c_max <= 1.05) else "❌ LỖI (Ngoài khoảng 0-1)"
            print(f"    - {col:<15}: Min={c_min:>8.4f}, Max={c_max:>8.4f}  {status}")
    
    # 3. Kiểm tra tính hiện hữu của cột Raw (Dùng cho Env)
    print("\n[3] Kiểm tra cột vật lý (Dùng cho Simulation):")
    PHYSICAL_COLS = ["queue_size", "time_to_deadline", "gas_t"]
    for col in PHYSICAL_COLS:
        if col in df.columns:
            print(f"    - {col:<15}: Min={df[col].min():>8.2f}, Max={df[col].max():>8.2f} ✓")
        else:
            print(f"    - {col:<15}: ❌ THIẾU")

    # 4. Kiểm tra phân phối Action (Cực kỳ quan trọng cho V24)
    print("\n[4] Kiểm tra phân phối Action (Action Distribution):")
    action_counts = df["action"].value_counts(normalize=True).sort_index()
    for action, pct in action_counts.items():
        bar = "█" * int(pct * 50)
        print(f"    - Action {int(action)}: {pct:>6.2%} {bar}")
    
    # 5. Kiểm tra chéo (Cross-check Axis integrity)
    if "gas_t" in df.columns and "s_gas_t0" in df.columns:
        print("\n[5] Kiểm tra chéo (Cross-check Axis integrity):")
        sample = df.sample(min(5, len(df)))
        print("    (Kiểm tra xem s_queue có tỷ lệ thuận với queue_size không)")
        cols_to_show = ["step_index", "action", "reward", "queue_size", "s_queue"]
        if "policy_type" in df.columns: cols_to_show.append("policy_type")
        print(sample[cols_to_show])

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "Final_Project/Data/transitions_discrete_v28.parquet"
    if Path(path).exists():
        check_sanity(path)
    else:
        print(f"Không tìm thấy file: {path}")
