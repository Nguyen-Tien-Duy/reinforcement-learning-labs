import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.offline_rl.schema import STATE_COLS

def validate_dataset(parquet_path: str):
    print("=" * 60)
    print(f"🔍 BẮT ĐẦU SIÊU ÂM DATASET: {Path(parquet_path).name}")
    print("=" * 60)

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"❌ Không thể đọc file. Lỗi: {e}")
        return

    # 1. KIỂM TRA NAN / INF
    print("\n1. KIỂM TRA TÍNH TOÀN VẸN (NAN/INF):")
    has_nan = df.isna().any().any()
    print(f"   - Có chứa NaN không? {'❌ CÓ (Tệ)' if has_nan else '✅ KHÔNG'}")
    
    # Check Infinity
    obs_sample = df[STATE_COLS].to_numpy()
    has_inf = np.isinf(obs_sample).any()
    print(f"   - State có bị vô cực (Infinity)? {'❌ CÓ (Tệ)' if has_inf else '✅ KHÔNG'}")

    # 2. KIỂM TRA PHÂN PHỐI REWARD & PENALTY
    print("\n2. KIỂM TRA REWARD (CHI PHÍ VÀ PHẠT):")
    r_min, r_max = df['reward'].min(), df['reward'].max()
    r_mean = df['reward'].mean()
    print(f"   - Reward Nhỏ nhất (Max Penalty): {r_min:,.2f}")
    if r_min < -3000:
        print("   ✅ CHUẨN MỰC! Đã thấy hình phạt 5 Tỷ Gwei kích hoạt (-5000+ pts).")
    else:
        print("   ❌ CẢNH BÁO: Không có án phạt khổng lồ nào trong data!")
    print(f"   - Reward Trung bình: {r_mean:,.2f}")

    # 3. KIỂM TRA TÍNH NHÂN QUẢ (CORRELATION)
    print("\n3. KIỂM TRA TÍNH NHÂN QUẢ (ORACLE EFFECT):")
    corr_queue = df['action'].corr(df['queue_size'])
    corr_time = df['action'].corr(df['time_to_deadline'])
    
    print(f"   - Tương quan (Action vs Queue): {corr_queue:+.4f}")
    if corr_queue > 0.15:
        print("   ✅ CHUẨN! AI đã biết: Hàng chờ đông (Queue tăng) -> Cần xả (Action tăng).")
    else:
        print("   ❌ CẢNH BÁO: Không có tương quan logic (Bị vỡ Proxy).")

    print(f"   - Tương quan (Action vs Time): {corr_time:+.4f}")
    if corr_time < -0.15:
        print("  CHUẨN! AI đã biết: Áp lực Deadline cao (Thời gian còn ít) -> Cần xả gấp (Action tăng).")
    
    # 4. TRỰC QUAN HÓA (CHỤP X-QUANG)
    print("\n Đang vẽ đồ thị chụp X-Quang xuất ra file png...")
    plt.figure(figsize=(18, 5))
    
    # Plot 1: Action Histogram
    plt.subplot(1, 3, 1)
    plt.hist(df['action'], bins=50, color='royalblue', edgecolor='white', alpha=0.8)
    plt.title(r'Action Distribution ($a_t \in [0, 1]$)')
    plt.xlabel('Tỷ lệ xả hàng')
    plt.ylabel('Tần suất')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 2: Reward Histogram (Log scale)
    plt.subplot(1, 3, 2)
    plt.hist(df['reward'], bins=50, color='firebrick', log=True, edgecolor='white', alpha=0.8)
    plt.title('Reward Logic (Log Scale)')
    plt.xlabel('Reward (Đã Normalization)')
    plt.ylabel('Tần suất (Log)')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 3: 2D Hexbin Action vs Queue
    plt.subplot(1, 3, 3)
    plt.hexbin(df['queue_size'], df['action'], gridsize=30, cmap='Blues', bins='log')
    plt.title('Tính Nhân Quả: Action vs Queue Size')
    plt.xlabel('Hàng chờ (Queue Size)')
    plt.ylabel('Hành động xả (Action)')
    plt.colorbar(label='log10(count)')
    
    out_img = "v3_validation_report.svg"
    plt.tight_layout()
    plt.savefig(out_img, dpi=120)
    plt.close()
    
    # ---------------------------------------------------------
    # 5. CHỨNG MINH TÍNH NHÂN QUẢ 11 TRƯỜNG STATE (LUẬN VĂN)
    # ---------------------------------------------------------
    print("\n🔬 Đang trích xuất 11 trường State để vẽ báo cáo Nhân quả...")
    
    # Extract state array to columns (directly from named semantic columns)
    df['state_p_t'] = df['s_congestion']  # p_t Congestion
    df['state_m_t'] = df['s_momentum']    # m_t Momentum
    
    plt.figure(figsize=(18, 5))
    
    # Plot 1: Action vs Time to Deadline
    plt.subplot(1, 3, 1)
    plt.hexbin(df['time_to_deadline'], df['action'], gridsize=30, cmap='Oranges', bins='log')
    plt.title('Trụ cột 2: Cạn thời gian ép xả bù\n(Time to Deadline vs Action)')
    plt.xlabel('Thời gian đến Deadline (Giờ)')
    plt.ylabel('Hành động xả (Action)')
    plt.gca().invert_xaxis() # Lật ngược trục X (Thời gian giảm dần từ trái sang phải)
    plt.colorbar(label='log10(count)')
    
    # Plot 2: Gas Momentum vs Action
    plt.subplot(1, 3, 2)
    plt.hexbin(df['state_m_t'], df['action'], gridsize=30, cmap='Purples', bins='log')
    plt.title('Trụ cột 1: Gia tốc giá (Momentum)\n(m_t âm -> Giữ lại, m_t dương -> Xả)')
    plt.xlabel('Gas Momentum (Khử chuẩn hóa)')
    plt.ylabel('Hành động xả (Action)')
    plt.colorbar(label='log10(count)')

    # Plot 3: Congestion vs Action
    plt.subplot(1, 3, 3)
    plt.hexbin(df['state_p_t'], df['action'], gridsize=30, cmap='Greens', bins='log')
    plt.title('Trụ cột 1: Áp lực mạng (Congestion)\n(Mạng Tắc nghẽn p_t -> Ép xả trước bão)')
    plt.xlabel('Áp lực mạng (p_t)')
    plt.ylabel('Hành động xả (Action)')
    plt.colorbar(label='log10(count)')
    
    out_img2 = "v3_state_causality_report.svg"
    plt.tight_layout()
    plt.savefig(out_img2, dpi=120)
    plt.close()

    print(f"✅ Hoàn tất! Hãy mở 2 file:")
    print(f"   1. {out_img} (X-Quang Hàng đợi & Trừng phạt)")
    print(f"   2. {out_img2} (Báo cáo Luận văn: Nhân quả của State)")

if __name__ == "__main__":
    DATASET_PATH = "/mnt/WindowsD/Reinforcement Learning/labs/Final_Project/Data/transitions_discrete_v9.parquet"
    validate_dataset(DATASET_PATH)
