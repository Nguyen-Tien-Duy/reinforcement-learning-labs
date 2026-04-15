import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def prove_causal_collapse():
    data_path = "/mnt/WindowsD/Reinforcement Learning/labs/Final_Project/Data/transitions_oracle_v4_named.parquet"
    print(f"Reading dataset: {data_path}")
    
    # Load a sample to keep it fast
    df = pd.read_parquet(data_path, columns=["action", "s_congestion", "episode_id"])
    sample_df = df.sample(n=min(100000, len(df)), random_state=42)
    
    # 1. Calculate Correlation
    corr = sample_df["action"].corr(sample_df["s_congestion"])
    print(f"Pearson Correlation between action and s_congestion: {corr:.4f}")
    
    # 2. Plotting
    plt.figure(figsize=(10, 6), dpi=120)
    
    # Use hexbin to match the user's plot style and show density
    plt.hexbin(sample_df["s_congestion"], sample_df["action"], gridsize=50, cmap='Greens', bins='log')
    plt.colorbar(label='log10(count)')
    
    # Plot the theoretical line: action = (congestion + 1) / 2
    x_theory = np.linspace(sample_df["s_congestion"].min(), sample_df["s_congestion"].max(), 100)
    y_theory = (x_theory + 1) / 2
    plt.plot(x_theory, y_theory, 'r--', alpha=0.8, label='Theoretical: action = (p_t+1)/2')
    
    plt.xlabel("Congestion (s_congestion)")
    plt.ylabel("Action (ground truth)")
    plt.title(f"Causal Collapse Proof\nCorrelation: {corr:.4f}")
    plt.legend()
    plt.grid(alpha=0.3)
    
    save_path = "/mnt/WindowsD/Reinforcement Learning/labs/Final_Project/causal_collapse_proof.png"
    plt.savefig(save_path)
    print(f"Proof plot saved to: {save_path}")

    # 3. Analyze the 'spread'
    # Calculate the residual from the linear rule
    sample_df["theoretical_action"] = (sample_df["s_congestion"] + 1) / 2
    sample_df["residual"] = (sample_df["action"] - sample_df["theoretical_action"]).abs()
    
    on_line_pct = (sample_df["residual"] < 0.01).mean() * 100
    print(f"Percentage of data points EXACTLY on the theoretical line (+/- 0.01): {on_line_pct:.2f}%")

if __name__ == "__main__":
    prove_causal_collapse()
