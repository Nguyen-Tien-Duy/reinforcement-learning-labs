import pandas as pd
import numpy as np
import sys

def check_miss_rate(file_path):
    print(f"[+] Đang kiểm tra Oracle trong file: {file_path} ...")
    try:
        df = pd.read_parquet(file_path)
        
        if 'policy_type' in df.columns:
            print("[i] Detected policy_type column. Filtering for Oracle (type=1) only...")
            df = df[df['policy_type'] == 1].copy()
        
        last_steps = df[df['done'] == 1].copy()
        
        action_bins = [0.0, 0.25, 0.5, 0.75, 1.0]
        exec_cap = 500.0
        
        ratios = last_steps['action'].map(lambda a: action_bins[int(a)])
        pre_exec_q = last_steps['queue_size'].values
        executed = np.minimum(np.floor(ratios.values * pre_exec_q), exec_cap)
        remaining_after_exec = pre_exec_q - executed
        
        remaining_after_exec[remaining_after_exec < 1.0] = 0.0
        remaining_after_exec = np.maximum(0, remaining_after_exec)
        
        misses = (remaining_after_exec > 0).sum()
        total = len(last_steps)
        
        miss_rate = (misses / total) * 100
        avg_q_after = remaining_after_exec.mean()
        max_q_after = remaining_after_exec.max()
        
        print("\n" + "="*50)
        print(f"BÁO CÁO ORACLE SOLVABILITY (POST-EXECUTION):")
        print(f" - Tổng số Episode      : {total}")
        print(f" - Số lần trễ hạn (Miss): {misses}")
        print(f" - TỈ LỆ MISS RATE      : {miss_rate:.2f}%")
        print(f" - Hàng đợi dư (Avg)    : {avg_q_after:.2f} txs")
        print(f" - Hàng đợi dư (Max)    : {max_q_after:.2f} txs")
        
        if misses > 0:
            print("\n[!] CHI TIẾT CÁC EPISODE BỊ SÓT HÀNG:")
            miss_mask = remaining_after_exec > 0
            missed_eps = last_steps[miss_mask].copy()
            missed_eps['leftover'] = remaining_after_exec[miss_mask]
            
            for _, row in missed_eps.iterrows():
                print(f"  - Episode {int(row['episode_id'])}: Còn dư {row['leftover']:.1f} txs (Tổng Q: {row['queue_size']:.1f})")
        
        print("="*50)
        
        if miss_rate == 0:
            print("🎉 TUYỆT VỜI: Oracle đã giải tối ưu 100.0%!")
        else:
            print(f"👍 GẦN TỐI ƯU: Chỉ còn rất ít ({misses}) episode bị sót.")
            
        print(f"\n  Action phân bố ở bước cuôi:")
        for a in sorted(last_steps['action'].unique()):
            pct = (last_steps['action'] == a).mean() * 100
            print(f"    Action {int(a)}: {pct:.1f}%")
            
    except Exception as e:
        print(f"[!] Lỗi khi đọc file: {e}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else 'Final_Project/Data/transitions_discrete_v25.parquet'
    check_miss_rate(path)
