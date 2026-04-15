import pandas as pd
import numpy as np
import sys

def check_miss_rate(file_path):
    print(f"[+] Đang kiểm tra Oracle trong file: {file_path} ...")
    try:
        df = pd.read_parquet(file_path)
        
        # Lọc ra các bước cuối cùng của mỗi episode (nơi xảy ra Deadline)
        last_steps = df[df['done'] == 1].copy()
        
        # s_queue là hàng đợi TRƯỚC KHI thực thi action.
        # Để tính hàng đợi THỰC SỰ sau khi thực thi, ta cần:
        #   remaining = max(0, s_queue - min(floor(ratio * s_queue), exec_cap))
        action_bins = [0.0, 0.25, 0.5, 0.75, 1.0]
        exec_cap = 500.0  # Must match training config
        
        ratios = last_steps['action'].map(lambda a: action_bins[int(a)])
        executed = np.minimum(
            np.floor(ratios.values * last_steps['s_queue'].values), 
            exec_cap
        )
        remaining_after_exec = last_steps['s_queue'].values - executed
        # Clear fractional remainders
        remaining_after_exec[remaining_after_exec < 1.0] = 0.0
        remaining_after_exec = np.maximum(0, remaining_after_exec)
        
        # Một episode bị coi là 'Miss' nếu hàng đợi SAU KHI thực thi > 0
        misses = (remaining_after_exec > 0).sum()
        total = len(last_steps)
        
        miss_rate = (misses / total) * 100
        avg_q_before = last_steps['s_queue'].mean()
        avg_q_after = remaining_after_exec.mean()
        max_q_after = remaining_after_exec.max()
        
        print("="*50)
        print(f"BÁO CÁO ORACLE SOLVABILITY (POST-EXECUTION):")
        print(f" - Tổng số Episode      : {total}")
        print(f" - Hàng đợi TB TRƯỚC xả : {avg_q_before:.2f}")
        print(f" - Hàng đợi TB SAU xả   : {avg_q_after:.2f}")
        print(f" - Hàng đợi Max SAU xả  : {max_q_after:.2f}")
        print(f" - Số lần trễ hạn       : {misses}")
        print(f" - TỈ LỆ MISS RATE      : {miss_rate:.2f}%")
        print("="*50)
        
        if miss_rate == 0:
            print("🎉 TUYỆT VỜI: Oracle đã giải tối ưu 100%!")
        elif miss_rate < 5:
            print(f"👍 GẦN TỐI ƯU: Chỉ còn {misses} episode bị sót.")
        else:
            print("⚠️ CẢNH BÁO: Oracle vẫn còn để sót hàng.")
            
        # Action distribution at last step
        print(f"\n  Action phân bố ở bước cuối:")
        for a in sorted(last_steps['action'].unique()):
            pct = (last_steps['action'] == a).mean() * 100
            print(f"    Action {int(a)}: {pct:.1f}%")
            
    except Exception as e:
        print(f"[!] Lỗi khi đọc file: {e}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else 'Final_Project/Data/transitions_discrete_v20.parquet'
    check_miss_rate(path)
