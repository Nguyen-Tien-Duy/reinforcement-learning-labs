import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_episode_behavior(file_path, episode_id):
    df = pd.read_parquet(file_path)
    ep_df = df[df['episode_id'] == episode_id].sort_values('step_index')
    
    if len(ep_df) == 0:
        print(f"Episode {episode_id} not found.")
        return

    fig, ax1 = plt.subplots(figsize=(15, 8))

    # Plot Gas Price (Trục trái)
    ax1.set_xlabel('Step Index')
    ax1.set_ylabel('Gas Price (Gwei)', color='tab:red')
    # Lưu ý: RAW mode, chia 1e9
    ax1.plot(ep_df['step_index'], ep_df['gas_t'] / 1e9, color='tab:red', alpha=0.6, label='Gas Price')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Plot Queue Size (Trục phải)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Queue Size', color='tab:blue')
    ax2.plot(ep_df['step_index'], ep_df['queue_size'], color='tab:blue', alpha=0.4, label='Queue Size')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Plot Actions (Các điểm chấm)
    # Action 4: Đỏ đậm, Action 0: Xám, Action 1-3: Xanh lá
    colors = {0: 'gray', 1: 'lime', 2: 'green', 3: 'teal', 4: 'black'}
    for a in range(5):
        mask = ep_df['action'] == a
        ax1.scatter(ep_df.loc[mask, 'step_index'], (ep_df.loc[mask, 'gas_t'] / 1e9) * 1.05, 
                    color=colors[a], label=f'Action {a}' if a in [0, 4] else None, s=20)

    plt.title(f'Behavior Analysis - Episode {episode_id} (beta=0.01, alpha=2.0)')
    fig.tight_layout()
    save_path = f'Final_Project/visualize/behavior_ep_{episode_id}.png'
    plt.savefig(save_path)
    print(f"[✓] Analysis plot saved to {save_path}")

    # In thống kê nhanh
    print("\nSTATISTICS:")
    print(f"- Total Steps: {len(ep_df)}")
    print(f"- Action Distribution:\n{ep_df['action'].value_counts(normalize=True).sort_index()}")
    
    # Kiểm tra xem có hiện tượng "Gom hàng xả cục" không
    action_4_indices = ep_df[ep_df['action'] == 4]['step_index'].values
    if len(action_4_indices) > 1:
        gaps = np.diff(action_4_indices)
        print(f"- Avg steps between Action 4: {gaps.mean():.2f}")

if __name__ == "__main__":
    analyze_episode_behavior('Final_Project/Data/transitions_v33_Patient_RAW.parquet', 1013)
