import pandas as pd
import numpy as np
import sys
import os

def audit_oracle_performance(file_path):
    print(f"[*] Đang phân tích hiệu quả kinh tế Oracle: {file_path}")
    try:
        df = pd.read_parquet(file_path)
        expert = df[df['policy_type'] == 1].copy()
        
        # 1. Tính toán chi phí Oracle thực tế (Physical Gwei)
        # Lưu ý: Trong RAW mode, gas_t là Wei, cần chia 1e9
        gas_prices_gwei = expert['gas_t'].values / 1e9
        
        # Action map cho 5 bins (0, 0.25, 0.5, 0.75, 1.0) * execution_capacity (500)
        action_bins = [0.0, 0.25, 0.5, 0.75, 1.0]
        exec_cap = 500.0
        
        # Tính Volume thực tế Oracle đã xả
        ratios = expert['action'].map(lambda a: action_bins[int(a)]).values
        pre_q = expert['queue_size'].values
        oracle_volume = np.minimum(np.floor(ratios * pre_q), exec_cap)
        oracle_cost = oracle_volume * gas_prices_gwei
        
        # 2. Giả lập chiến thuật Greedy (Xả 100% ngay khi có thể)
        # Greedy sẽ xả min(queue, 500) tại mỗi bước
        greedy_volume = np.zeros_like(oracle_volume)
        current_greedy_q = 0.0
        greedy_costs = []
        
        # Chúng ta cần mô phỏng lại từng episode cho Greedy
        for ep_id, ep_df in expert.groupby('episode_id'):
            arrivals = ep_df['transaction_count'].values * 0.5 # arrival_scale
            gas = ep_df['gas_t'].values / 1e9
            q = 0.0
            ep_greedy_cost = 0.0
            for t in range(len(ep_df)):
                q += arrivals[t]
                exec_v = min(q, exec_cap)
                ep_greedy_cost += exec_v * gas[t]
                q -= exec_v
            greedy_costs.append(ep_greedy_cost)

        # 3. Tổng hợp kết quả
        total_oracle_cost = oracle_cost.sum()
        total_greedy_cost = sum(greedy_costs)
        savings = total_greedy_cost - total_oracle_cost
        savings_pct = (savings / total_greedy_cost) * 100 if total_greedy_cost > 0 else 0
        
        print("\n" + "="*50)
        print("BÁO CÁO HIỆU QUẢ KINH TẾ (ORACLE VS GREEDY)")
        print("="*50)
        print(f"💰 Tổng chi phí Greedy : {total_greedy_cost:,.2f} Gwei")
        print(f"💰 Tổng chi phí Oracle : {total_oracle_cost:,.2f} Gwei")
        print(f"🚀 Số tiền tiết kiệm   : {savings:,.2f} Gwei")
        print(f"📈 Tỷ lệ giảm chi phí  : {savings_pct:.2f}%")
        print("="*50)
        
        if savings_pct > 20:
            print("🎉 KẾT LUẬN: Oracle hoạt động cực kỳ hiệu quả, vượt xa chiến thuật truyền thống!")
        else:
            print("⚠️ CẢNH BÁO: Mức tiết kiệm thấp, kiểm tra lại biến động giá Gas trong dữ liệu.")

    except Exception as e:
        print(f"[!] Lỗi: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audit_oracle_performance(sys.argv[1])
    else:
        print("Cách dùng: python audit_oracle_savings.py <path_to_parquet>")
