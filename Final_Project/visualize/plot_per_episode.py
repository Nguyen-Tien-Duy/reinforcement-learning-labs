"""
Plot Per-Episode Comparison: CQL + Safety 20% vs Greedy
Vẽ scatter plot và histogram savings trên toàn bộ 242 episodes test.

Usage:
    python Final_Project/visualize/plot_per_episode.py \
        --data Final_Project/Data/transitions_v33_L2_Batching_RAW.parquet \
        --model d3rlpy_logs/DiscreteCQL_V6_20260428_0426_20260428042630/model_160000.d3 \
        --output report/Images/06_per_episode_comparison.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import d3rlpy

sys.path.append(str(Path(__file__).resolve().parents[1] / "code"))
from utils.offline_rl.enviroment import CharityGasEnv
from utils.offline_rl.config import TransitionBuildConfig

# ── Palette ───────────────────────────────────────────────────────────────────
BG = "#0F1117"; PANEL = "#1A1D27"; TEXT = "#E2E8F0"
BLUE = "#4C9BE8"; ORANGE = "#F4956A"; GREEN = "#50C878"; RED = "#FF6B6B"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": "#2D3148", "axes.labelcolor": TEXT,
    "text.color": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    "grid.color": "#2A2D3E", "grid.alpha": 0.5,
    "font.family": "DejaVu Sans", "font.size": 10,
})

SAFETY_THRESHOLD = 0.20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="Final_Project/Data/transitions_v33_L2_Batching_RAW.parquet")
    parser.add_argument("--model",
                        default="d3rlpy_logs/DiscreteCQL_V6_20260428_0426_20260428042630/model_160000.d3")
    parser.add_argument("--output",
                        default="report/Images/06_per_episode_comparison.png")
    args = parser.parse_args()

    config = TransitionBuildConfig()
    df = pd.read_parquet(args.data)
    unique_eps = sorted(df["episode_id"].unique())
    test_ids = unique_eps[int(len(unique_eps) * 0.8):]
    ep_list = [d.reset_index(drop=True)
               for _, d in df[df["episode_id"].isin(test_ids)].groupby("episode_id")]
    del df

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = d3rlpy.load_learnable(args.model, device=device)
    print(f"[+] Loaded model on {device}, evaluating {len(ep_list)} episodes...")

    g_costs, c_costs = [], []
    for i, ep_df in enumerate(ep_list):
        if (i + 1) % 50 == 0:
            print(f"   {i+1}/{len(ep_list)}...")

        # Greedy
        env = CharityGasEnv(ep_df, config, mins=None, maxs=None)
        env.reset(); done = False; gc = 0.0
        while not done:
            _, _, t, tr, info = env.step(4)
            gc += info.get("cost", 0.0)
            if t or tr: done = True
        g_costs.append(gc)

        # CQL + Safety
        env = CharityGasEnv(ep_df, config, mins=None, maxs=None)
        obs, _ = env.reset(); done = False; cc = 0.0
        while not done:
            tr_ratio = env.time_to_deadline / env.config.episode_hours
            if tr_ratio < SAFETY_THRESHOLD:
                action = 4
            else:
                obs_c = np.ascontiguousarray(obs.reshape(1, -1), dtype=np.float32)
                action = int(model.predict(obs_c)[0])
            obs, _, t, tr, info = env.step(action)
            cc += info.get("cost", 0.0)
            if t or tr: done = True
        c_costs.append(cc)

    g = np.array(g_costs)
    c = np.array(c_costs)
    savings = (g - c) / g * 100

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("CQL + Safety 20% vs Greedy — Per-Episode Analysis "
                 f"({len(ep_list)} eps)",
                 fontsize=14, color=TEXT, fontweight="bold")

    # Scatter
    ax = axes[0]
    colors = [GREEN if s > 0 else RED for s in savings]
    ax.scatter(g, c, c=colors, s=15, alpha=0.7, edgecolors="none")
    lim = max(g.max(), c.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="#888", lw=1, label="Break-even")
    ax.set_xlabel("Greedy Gas Cost", fontsize=11)
    ax.set_ylabel("CQL + Safety Gas Cost", fontsize=11)
    ax.set_title("Per-Episode Cost Comparison", fontsize=12, color=BLUE)
    ax.legend(fontsize=9, framealpha=0.3)
    ax.grid(True, lw=0.4)
    win_rate = (savings > 0).sum() / len(savings) * 100
    ax.text(0.05, 0.95,
            f"Win Rate: {win_rate:.0f}% ({(savings>0).sum()}/{len(savings)})",
            transform=ax.transAxes, fontsize=10, color=GREEN,
            va="top", fontweight="bold")

    # Histogram
    ax2 = axes[1]
    ax2.hist(savings, bins=40, color=BLUE, alpha=0.8, edgecolor="none")
    ax2.axvline(savings.mean(), color=ORANGE, lw=2, ls="--",
                label=f"Mean: {savings.mean():.1f}%")
    ax2.axvline(0, color=RED, lw=1.5, ls="-", alpha=0.5, label="Break-even")
    ax2.set_xlabel("Savings vs Greedy (%)", fontsize=11)
    ax2.set_ylabel("Number of Episodes", fontsize=11)
    ax2.set_title("Distribution of Gas Savings", fontsize=12, color=ORANGE)
    ax2.legend(fontsize=9, framealpha=0.3)
    ax2.grid(True, lw=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    print(f"\n[✓] Saved {args.output}")
    print(f"Win Rate: {win_rate:.1f}% | Mean: {savings.mean():.1f}% | "
          f"Median: {np.median(savings):.1f}%")
    print(f"Min: {savings.min():.1f}% | Max: {savings.max():.1f}%")


if __name__ == "__main__":
    main()
