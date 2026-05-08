import pandas as pd
import numpy as np
import sys

def check_miss_rate(file_path):
    print(f"[+] Đang thẩm định năng lực Oracle trên file: {file_path}")
    try:
        df = pd.read_parquet(file_path)
        
        if 'policy_type' not in df.columns:
            print("❌ File không có cột policy_type. Không thể check Oracle.")
            return

        # 1. Bốc TOÀN BỘ episodes Expert (nhãn 1) để đánh giá "Đỉnh cao" của Oracle
        df_expert = df[df['policy_type'] == 1].copy()
        
        # 2. Xác định ranh giới 10% cuối (Tập Test) để báo cáo thêm
        all_unique_ids = sorted(df['episode_id'].unique())
        split_idx = int(len(all_unique_ids) * 0.8)
        test_ids = all_unique_ids[split_idx:]
        
        def calculate_stats(data_df):
            last_steps = data_df[data_df['done'] == 1].copy()
            if len(last_steps) == 0: return None
            
            action_bins = [0.0, 0.25, 0.5, 0.75, 1.0]
            exec_cap = 500.0
            ratios = last_steps['action'].map(lambda a: action_bins[int(a)])
            pre_exec_q = last_steps['queue_size'].values
            executed = np.minimum(np.floor(ratios.values * pre_exec_q), exec_cap)
            remaining = np.maximum(0, pre_exec_q - executed)
            remaining[remaining < 1.0] = 0.0
            
            misses = (remaining > 0).sum()
            return {
                "total": len(last_steps),
                "misses": misses,
                "rate": (misses / len(last_steps)) * 100,
                "avg_q": remaining.mean(),
                "max_q": remaining.max()
            }

        # Tính toán cho Toàn bộ Oracle
        overall = calculate_stats(df_expert)
        # Tính toán riêng cho Oracle trong tập Test
        df_test_expert = df_expert[df_expert['episode_id'].isin(test_ids)]
        test_stats = calculate_stats(df_test_expert)

        print("\n" + "="*50)
        print(f"BÁO CÁO NĂNG LỰC ORACLE (EXPERT ONLY):")
        if overall:
            print(f" 🏆 TỔNG THỂ ({overall['total']} eps): MISS RATE = {overall['rate']:.2f}% | Avg Q = {overall['avg_q']:.2f} txs")
        if test_stats:
            print(f" 🎯 TẬP TEST  ({test_stats['total']} eps): MISS RATE = {test_stats['rate']:.2f}% | Max Q = {test_stats['max_q']:.2f} txs")
        print("="*50)
        
        if overall and overall['rate'] == 0:
            print("🎉 XÁC NHẬN: Thuật toán Oracle DP đạt độ tối ưu TUYỆT ĐỐI (0% Miss)!")
        elif overall:
            print(f"⚠️ CẢNH BÁO: Oracle vẫn bị sót {overall['misses']} episodes. Kiểm tra lại beta/penalty.")
            
    except Exception as e:
        print(f"[!] Lỗi khi xử lý: {e}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else 'Final_Project/Data/transitions_discrete_v28.parquet'
    check_miss_rate(path)
