import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Chẩn đoán Offline RL Dataset")
    parser.add_argument("--data", type=str, default="../Data/data_2024-04-10_2026-04-10.parquet", help="Path to parquet file")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[!] Không tìm thấy file: {data_path.resolve()}")
        print("Hãy chạy script tại thư mục code hoặc chỉ định đường dẫn đúng thông qua tham số --data")
        return

    print(f"[*] Đang nạp dữ liệu từ: {data_path.name}...")
    df = pd.read_parquet(data_path)
    print(f"[+] Đã tải {len(df):,} dòng dữ liệu.")
    
    required_cols = ['action', 'time_to_deadline', 'queue_size', 'reward']
    missing = [c for c in required_cols if c not in df.columns]
    
    # Nếu data là Raw Data chưa build transitions, ta tự động build luôn để vẽ
    if missing:
        print(f"[!] Dữ liệu đang thiếu các cột Transition: {missing}.")
        print("[*] Đang tự động chạy Transition Builder (Chuyển đổi dữ liệu thô sang RL Data)...")
        import sys
        sys.path.append(str(Path(__file__).resolve().parent))
        from utils.offline_rl.config import TransitionBuildConfig
        from utils.offline_rl.transition_builder import build_transitions
        
        # Build với mặc định
        config = TransitionBuildConfig()
        df = build_transitions(df, config=config, use_oracle=False)
        print("[+] Xây dựng Transition thành công!")
        
    print("[*] Đang tính toán và vẽ biểu đồ...")
    plt.figure(figsize=(16, 12))
    plt.style.use('bmh') # Style matplotlib cho dễ nhìn
    # Cấu hình font chữ cơ bản để hỗ trợ tiếng Việt (tuỳ hệ thống)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Tahoma', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
    
    # --- PLOT 1: Action vs Time to Deadline ---
    plt.subplot(2, 2, 1)
    
    # Lấy bins cho deadline (đảm bảo nó dương từ 0 đến max time)
    max_time = min(24, int(df['time_to_deadline'].max()))
    bins = np.arange(0, max_time + 1, 1)
    # Lật ngược time_to_deadline để vẽ (từ trái qua phải là từ lúc đầy đến lúc cạn)
    df['deadline_bin'] = pd.cut(df['time_to_deadline'], bins=bins)
    action_by_time = df.groupby('deadline_bin', observed=True)['action'].mean()
    
    # Vẽ dạng cột
    x_labels = [f"{b.left}-{b.right}h" for b in action_by_time.index]
    plt.bar(x_labels, action_by_time.values, color='coral', edgecolor='black')
    
    plt.title("BÀI TEST 1: Sự vô cảm của Action theo Thời gian")
    plt.xlabel("Khoảng thời gian CÒN LẠI đến Deadline")
    plt.ylabel("Xác suất / Giá trị Hành động trung bình")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # --- PLOT 2: Reward Distribution (Kiểm toán hình phạt) ---
    plt.subplot(2, 2, 2)
    rewards = df['reward'].dropna()
    plt.hist(rewards, bins=50, color='crimson', log=True, edgecolor='black')
    plt.title("BÀI TEST 2: Lịch sử 'Bị phạt' trong Data (Log Scale)")
    plt.xlabel("Trị giá Reward")
    plt.ylabel("Tần suất (Log)")
    
    # Vẽ đưòng cắm mốc Reward bằng 0
    plt.axvline(0, color='blue', linestyle='dashed', linewidth=2)
    
    # --- PLOT 3: Action vs Queue Size ---
    plt.subplot(2, 2, 3)
    # Lấy mẫu 5000 điểm ngẫu nhiên để vẽ không bị quá lag
    sample_df = df.sample(min(5000, len(df)))
    
    plt.scatter(sample_df['queue_size'], sample_df['action'], alpha=0.2, color='teal', s=10)
    plt.title("BÀI TEST 3: Tương quan Hành động và Lượng Queue rác")
    plt.xlabel("Kích thước Queue đang ôm (Đơn vị Queue)")
    plt.ylabel("Chỉ số Hành động Action")
    
    # Thêm đường xu hướng (Trend line)
    z = np.polyfit(sample_df['queue_size'].values, sample_df['action'].values, 1)
    p = np.poly1d(z)
    plt.plot(sample_df['queue_size'], p(sample_df['queue_size']), "r--")
    
    # --- PLOT 4: BÁO CÁO SỐ LIỆU ---
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # Phân tích một chút số liệu
    missed_target = df[(df['time_to_deadline'] <= 0.1) & (df['queue_size'] > 0)]
    miss_count = len(missed_target)
    
    corr_action_queue = df['action'].corr(df['queue_size'])
    corr_action_time = df['action'].corr(df['time_to_deadline'])
    
    stats_text = (
        f"=== CHẨN ĐOÁN LÂM SÀNG DATASET ===\n\n"
        f"1. Tổng số Transitions: {len(df):,} mẫu\n\n"
        f"2. Phạm vi Reward: [Min: {rewards.min():,.2f} ... Max: {rewards.max():,.2f}]\n"
        f"   => Lý tưởng: Cần có điểm Min rất thấp phạt Deadline.\n\n"
        f"3. Số lần 'Lỡ tay' sát giờ (q>0, t~0): {miss_count:,} lần\n"
        f"   => Kì vọng: Rất thấp (Policy cũ sợ chết), hoặc bị phạt nặng.\n\n"
        f"4. Sức mạnh Nhân quả (Hệ số Tương quan Pearson):\n"
        f"   - Tương quan (Action , Queue) = {corr_action_queue:+.4f}\n"
        f"     (Lý tưởng: Càng lớn càng tốt - Lắm rác thì phải xả)\n\n"
        f"   - Tương quan (Action , T_Deadline) = {corr_action_time:+.4f}\n"
        f"     (Lý tưởng: Âm - Ít thời gian thì áp lực hành động lớn)\n\n"
        f"KẾT LUẬN: Nếu Tương quan quanh mức 0.00, Dataset của \n"
        f"bạn HOÀN TOÀN TỊT NGÒI về mặt Logic Hành động."
    )
    plt.text(0.05, 0.95, stats_text, fontsize=13, va='top', ha='left')
    
    out_path = Path("data_diagnostics.png").resolve()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\n[+] HOÀN THẤT! Hệ thống đã xuất bản vẽ ra file ảnh tại:\n    {out_path}")

if __name__ == "__main__":
    main()
