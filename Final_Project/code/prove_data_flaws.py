import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Cấu hình font hệ thống
plt.style.use('bmh')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Tahoma', 'DejaVu Sans', 'sans-serif']

def main():
    data_path = Path("../Data/transitions_hardened_v2.parquet")
    print(f"[*] Đang tải dữ liệu {data_path.name} để chứng minh...")
    df = pd.read_parquet(data_path)
    
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle('BẰNG CHỨNG THÉP: TẠI SAO DATA HIỆN TẠI KHẾN MODEL BỊ SỤP (POLICY COLLAPSE)', 
                 fontsize=18, fontweight='bold', color='darkred')

    # --- Plot 1: Action vs Time to Deadline (Correlation = 0) ---
    plt.subplot(2, 3, 1)
    
    # Nhóm Action trung bình theo các mốc thời gian còn lại (0h - 24h)
    bins = np.arange(0, 25, 1)
    df['time_bin'] = pd.cut(df['time_to_deadline'], bins=bins)
    
    grp = df.groupby('time_bin', observed=True)['action'].agg(['mean', 'std'])
    x_labels = [f"Còn {b.right}h" for b in grp.index]
    
    plt.errorbar(range(len(grp)), grp['mean'], yerr=grp['std'], fmt='-o', color='purple', ecolor='lightgray', capsize=5, linewidth=2)
    plt.axhline(df['action'].mean(), color='red', linestyle='--', label=f"Avg Baseline = {df['action'].mean():.2f}")
    
    plt.title("1. Sự vô cảm với Deadline\n(Đường đi ngang = Không có chiến thuật)", fontsize=13)
    plt.xlabel("Trục Thời Gian Lùi Về 0h")
    plt.ylabel("Action Trung Bình (kèm độ rung lắc)")
    plt.xticks(range(0, len(grp), 3), [x_labels[i] for i in range(0, len(grp), 3)], rotation=30)
    plt.legend()
    
    # --- Plot 2: Action vs Queue Size (Correlation = 0) ---
    plt.subplot(2, 3, 2)
    
    # Lấy mẫu ngẫu nhiên 100k điểm vẽ Hexbin (Mật độ)
    sample_df = df.sample(min(100000, len(df)))
    plt.hexbin(sample_df['queue_size'], sample_df['action'], gridsize=25, cmap='Blues', bins='log')
    plt.colorbar(label='Độ đậm đặc (Log Frequency)')
    
    # Trendline
    z = np.polyfit(sample_df['queue_size'], sample_df['action'], 1)
    p = np.poly1d(z)
    plt.plot(sample_df['queue_size'], p(sample_df['queue_size']), "r--", linewidth=3, label="Đường Xu Hướng")
    
    corr_val = df['action'].corr(df['queue_size'])
    plt.title(f"2. Sự ngẫu nhiên với Nợ (Queue)\n(Tương quan Pearson: {corr_val:.4f})", fontsize=13)
    plt.xlabel("Quy mô Hàng đợi (Queue Size)")
    plt.ylabel("Mức độ Xả (Action)")
    plt.legend()
    
    # --- Plot 3: Missing Reward Penalty ---
    plt.subplot(2, 3, 3)
    
    # Phân loại Success vs Miss
    # Chúng ta cho dãn một xíu (deadline <= 0.1) để bắt được các điểm gần sát
    df['missed'] = (df['time_to_deadline'] <= 0.1) & (df['queue_size'] > 0)
    
    # Vẽ Biểu đồ hộp bằng thuần Matplotlib
    miss_rewards = df[df['missed'] == True]['reward'].dropna()
    hit_rewards = df[df['missed'] == False]['reward'].dropna()
    
    bp = plt.boxplot([hit_rewards, miss_rewards], labels=['Không bị (Bình thường)', 'Có bị (Vỡ Deadline)'],
                     patch_artist=True, notch=True)
    
    # Tô màu thủ công
    colors = ['lightgreen', 'salmon']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    plt.title("3. Lỗ hổng về Penalty khi Vỡ Nợ\n(Chưa từng bị đánh đòn nặng!)", fontsize=13)
    plt.xlabel("Có Đọng Queue lúc hết giờ không?")
    plt.ylabel("Reward Nhận được (Kì vọng là âm vô cực)")
    plt.yscale('symlog') # Dùng symlog để vẽ số âm lớn
    
    # --- GIẢI THÍCH (Text Explanations) ---
    ax4 = plt.subplot(2, 1, 2)
    ax4.axis('off')
    
    text = (
        "GIẢI MÃ 3 BẰNG CHỨNG 'THÉP' (Sử dụng để phản biện Data Pipeline): \n\n"
        "🔥 Biểu đồ 1 [Trái]: Nếu AI có khái niệm về Thời gian, đường màu tím BẮT BUỘC phải ngóc đầu nhô cao lên khi thời gian chạy về 'Còn 1h'.\n"
        "   👉 Thực tế: Đường tím nằm im thin thít ở mốc 0.38 từ lúc 24h cho đến tận 1h cuối.\n"
        "      Kết luận: Cái 'Chuyên gia' vạch ra hành động trong Data này bị MÙ THỜI GIAN.\n\n"
        "🔥 Biểu đồ 2 [Giữa]: Nếu 'Nợ' thì phải 'Xả'. Queue càng lớn, Action phải càng cao (Đường xu hướng nét đứt màu đỏ dốc lên).\n"
        "   👉 Thực tế: Đường xu hướng màu đỏ NẰM NGANG SONG SONG TRỤC X, hệ số tương quan r = 0.0067.\n"
        "      Kết luận: Đại lượng Queue không đóng góp một chút logic nào vào việc tạo ra Action của Data.\n\n"
        "🔥 Biểu đồ 3 [Phải]: Cột False (Bình thường), Cột True (Vỡ Deadline nợ đọng). Nếu Penalty vỡ nợ là cực khủng, hộp True phải cắm sâu xuống Vực Thẳm.\n"
        "   👉 Thực tế: Trục Y của hộp True lơ lửng ở vài ngàn (không có -2.000.000). RL Agent là một sinh vật thực dụng, nó cân cái Reward phạt lèo tèo này\n"
        "      so với cái phí Gas đắt đỏ phải trả, nó thà 'Nằm im chịu trận, cắn Action hằng số 0' còn hơn là Xả khí Gas.\n\n"
        "⚠️ GIẢI PHÁP GỐC RỄ: Xóa bộ Data này đi. Vào file Transitions Builder, kích hoạt `USE_ORACLE=TRUE` để gán thuật toán Quy Hoạch Động (DP) sửa lại\n"
        "   toàn bộ cột Action, ĐỒNG THỜI fix code dòng logic Tính Phạt Hết Giờ để nó giáng một đòn chí mạng thực sự vào Model."
    )
    
    plt.text(0.01, 0.95, text, fontsize=14, va='top', ha='left', family='sans-serif', 
             bbox=dict(boxstyle="round,pad=1.5", fc="#FFF9E6", ec="darkorange", lw=2.5))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = Path("proof_logic.png").resolve()
    plt.savefig(out_path, dpi=120)
    print(f"\n[+] ĐÃ VẼ XONG BẰNG CHỨNG! File lưu tại: {out_path}")

if __name__ == '__main__':
    main()
