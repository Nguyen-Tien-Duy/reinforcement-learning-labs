import pandas as pd
import sys

def inspect_parquet(path):
    print(f"\n[*] Đang soi file: {path}")
    df = pd.read_parquet(path)
    
    print(f"\n[1] Kích thước: {df.shape[0]} dòng x {df.shape[1]} cột")
    
    print("\n[2] Danh sách tất cả các cột (37 cột):")
    cols = df.columns.tolist()
    for i in range(0, len(cols), 3):
        print(f"    {cols[i:i+3]}")
        
    print("\n[3] 5 dòng dữ liệu mẫu (Snapshot):")
    # Chọn một vài cột tiêu biểu để xem cho gọn
    display_cols = ["episode_id", "step_index", "queue_size", "s_queue", "action", "reward", "policy_type"]
    available_cols = [c for c in display_cols if c in df.columns]
    print(df[available_cols].head())

    print("\n[4] Kiểm tra giá trị policy_type (Oracle=1, Suboptimal=2, Behavioral=3):")
    if "policy_type" in df.columns:
        print(df["policy_type"].value_counts())
    else:
        print("❌ Không tìm thấy cột policy_type!")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "Final_Project/Data/transitions_discrete_v21.parquet"
    inspect_parquet(path)
