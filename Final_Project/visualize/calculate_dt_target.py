import pandas as pd
import numpy as np

def calculate_oracle_target_return():
    input_path = "Final_Project/Data/transitions_v33_L2_Batching_RAW.parquet"
    print(f"[*] Đang phân tích dữ liệu từ {input_path}...")
    df = pd.read_parquet(input_path)

    # Chia tách tập Train (80% đầu)
    unique_eps = sorted(df['episode_id'].unique())
    train_ids = unique_eps[:int(len(unique_eps) * 0.8)]
    train_df = df[df['episode_id'].isin(train_ids)]

    # Lọc 40% Episode tốt nhất (Expert/Oracle)
    ep_rewards = train_df.groupby('episode_id')['reward'].sum()
    threshold = ep_rewards.quantile(0.6)
    expert_rewards = ep_rewards[ep_rewards >= threshold]

    # Tính toán các chỉ số thống kê
    mean_return = expert_rewards.mean()
    max_return = expert_rewards.max()
    min_return = expert_rewards.min()
    median_return = expert_rewards.median()

    print("\n" + "="*50)
    print("📊 BÁO CÁO MỤC TIÊU CHO DECISION TRANSFORMER")
    print("="*50)
    print(f"Số lượng tập Expert: {len(expert_rewards)}")
    print(f"Lợi nhuận Cao nhất (Max): {max_return:.2f}")
    print(f"Lợi nhuận Trung bình (Mean): {mean_return:.2f}")
    print(f"Lợi nhuận Trung vị (Median): {median_return:.2f}")
    print(f"Lợi nhuận Thấp nhất (Min Expert): {min_return:.2f}")
    print("="*50)
    
    print(f"\n💡 LỜI KHUYÊN:")
    print(f"Hãy dùng con số [ {mean_return:.2f} ] làm Target Return khi đánh giá.")
    print(f"Nếu muốn AI 'vượt ngưỡng', hãy thử con số [ {max_return:.2f} ].")
    print("="*50)

if __name__ == "__main__":
    calculate_oracle_target_return()
