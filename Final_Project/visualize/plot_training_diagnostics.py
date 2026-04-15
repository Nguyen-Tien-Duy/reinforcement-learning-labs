#!/usr/bin/env python3
"""
Training Diagnostics for Discrete CQL.
Generates 5 diagnostic plots from d3rlpy training logs:

1. Loss Curves (Total, TD, Conservative) — Convergence proof
2. Gradient Norms per Layer — Stability proof (no explosion/vanishing)
3. Q-Value Analysis — Action preference proof (does AI prefer high actions?)
4. Action Distribution per Checkpoint — Learning evolution
5. Evaluation Summary — Miss rate & cost per checkpoint

Usage:
    python plot_training_diagnostics.py \
        --log-dir d3rlpy_logs/DiscreteCQL_V6_<timestamp>/ \
        --data Final_Project/Data/transitions_discrete_v17.parquet \
        [--eval-params ...]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def parse_args():
    p = argparse.ArgumentParser(description="CQL Training Diagnostics")
    p.add_argument("--log-dir", type=Path, required=True, help="d3rlpy log directory")
    p.add_argument("--data", type=Path, default=None, help="Transition parquet for Q-value analysis")
    p.add_argument("--output-dir", type=Path, default=None, help="Output directory for plots")
    return p.parse_args()


def plot_loss_curves(log_dir: Path, ax_total, ax_td, ax_cons):
    """Plot 1: Loss convergence curves."""
    for csv_name, ax, title, color in [
        ("loss.csv", ax_total, "Total Loss", "#e74c3c"),
        ("td_loss.csv", ax_td, "TD Loss", "#3498db"),
        ("conservative_loss.csv", ax_cons, "Conservative Loss", "#2ecc71"),
    ]:
        csv_path = log_dir / csv_name
        if not csv_path.exists():
            ax.set_title(f"{title} (NOT FOUND)")
            continue
        df = pd.read_csv(csv_path)
        steps = df.iloc[:, 0].values
        values = df.iloc[:, 1].values
        
        ax.plot(steps, values, color=color, alpha=0.3, linewidth=0.5)
        # Smoothed (rolling mean)
        window = max(1, len(values) // 50)
        smoothed = pd.Series(values).rolling(window, min_periods=1).mean().values
        ax.plot(steps, smoothed, color=color, linewidth=2, label=f"{title} (smoothed)")
        
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Annotate final value
        ax.annotate(f"Final: {values[-1]:.4f}", xy=(steps[-1], smoothed[-1]),
                   fontsize=9, ha="right", va="bottom", color=color,
                   fontweight="bold")


def plot_gradient_norms(log_dir: Path, ax):
    """Plot 2: Gradient norms per layer — detect explosion/vanishing."""
    grad_files = sorted(log_dir.glob("*_grad.csv"))
    if not grad_files:
        ax.set_title("Gradient Norms (NO DATA)")
        return
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(grad_files)))
    for gf, color in zip(grad_files, colors):
        df = pd.read_csv(gf)
        steps = df.iloc[:, 0].values
        values = df.iloc[:, 1].values
        
        # Simplify label
        label = gf.stem.replace("q_funcs.0.", "").replace("_grad", "")
        label = label.replace("._encoder._layers.", "L").replace(".weight", ".W").replace(".bias", ".B")
        
        window = max(1, len(values) // 30)
        smoothed = pd.Series(values).rolling(window, min_periods=1).mean().values
        ax.plot(steps, smoothed, color=color, linewidth=1.5, label=label)
    
    ax.set_title("Gradient Norms per Layer")
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")


def plot_q_value_analysis(log_dir: Path, data_path: Path, ax):
    """Plot 3: Q-values per action — does AI prefer the right actions?"""
    if data_path is None or not data_path.exists():
        ax.set_title("Q-Value Analysis (NO DATA)")
        return
    
    # Find latest model
    models = sorted(log_dir.glob("model_*.d3"), key=lambda p: int(p.stem.split("_")[1]))
    if not models:
        ax.set_title("Q-Value Analysis (NO MODELS)")
        return
    
    try:
        import d3rlpy
        algo = d3rlpy.load_learnable(str(models[-1]))
        
        # Sample states from data
        df = pd.read_parquet(data_path)
        from utils.offline_rl.schema import STATE_COLS
        states = df[STATE_COLS].sample(n=min(10000, len(df)), random_state=42).to_numpy(dtype=np.float32)
        
        # Get Q-values for all actions
        import torch
        with torch.no_grad():
            obs_tensor = torch.tensor(states, dtype=torch.float32)
            # Access internal Q-function
            q_func = algo._impl._q_funcs[0]
            q_values = q_func(obs_tensor).cpu().numpy()  # Shape: (N, 5)
        
        action_labels = ["A0 (0%)", "A1 (25%)", "A2 (50%)", "A3 (75%)", "A4 (100%)"]
        means = q_values.mean(axis=0)
        stds = q_values.std(axis=0)
        
        bars = ax.bar(range(5), means, yerr=stds, 
                      color=["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"],
                      alpha=0.8, capsize=5, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(5))
        ax.set_xticklabels(action_labels, fontsize=9)
        ax.set_title(f"Mean Q-Values per Action\n(Model: {models[-1].stem})")
        ax.set_ylabel("Q-Value")
        ax.grid(True, alpha=0.3, axis="y")
        
        # Annotate
        best = np.argmax(means)
        ax.annotate(f"★ Best: {action_labels[best]}", xy=(best, means[best]),
                   xytext=(best, means[best] + stds[best] * 0.5),
                   fontsize=10, ha="center", fontweight="bold", color="#27ae60")
        
    except Exception as e:
        ax.set_title(f"Q-Value Analysis\n(Error: {e})")


def plot_action_distribution_evolution(log_dir: Path, data_path: Path, ax):
    """Plot 4: How action distribution changes across checkpoints."""
    if data_path is None or not data_path.exists():
        ax.set_title("Action Evolution (NO DATA)")
        return
    
    models = sorted(log_dir.glob("model_*.d3"), key=lambda p: int(p.stem.split("_")[1]))
    if not models:
        ax.set_title("Action Evolution (NO MODELS)")
        return
    
    # Sample a smaller subset 
    try:
        import d3rlpy
        df = pd.read_parquet(data_path)
        from utils.offline_rl.schema import STATE_COLS
        states = df[STATE_COLS].sample(n=min(5000, len(df)), random_state=42).to_numpy(dtype=np.float32)
        
        # Pick ~6 checkpoints evenly spaced
        indices = np.linspace(0, len(models)-1, min(6, len(models)), dtype=int)
        selected = [models[i] for i in indices]
        
        action_dist_per_ckpt = {}
        for m in selected:
            algo = d3rlpy.load_learnable(str(m))
            actions = algo.predict(states)
            counts = np.bincount(actions.astype(int), minlength=5)
            action_dist_per_ckpt[m.stem] = counts / counts.sum() * 100
        
        # Stacked bar chart
        x = np.arange(len(selected))
        width = 0.6
        bottoms = np.zeros(len(selected))
        colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"]
        labels = ["A0 (0%)", "A1 (25%)", "A2 (50%)", "A3 (75%)", "A4 (100%)"]
        
        for a_id in range(5):
            vals = [action_dist_per_ckpt[m.stem][a_id] for m in selected]
            ax.bar(x, vals, width, bottom=bottoms, color=colors[a_id], label=labels[a_id], edgecolor="white", linewidth=0.5)
            bottoms += vals
        
        ax.set_xticks(x)
        step_labels = [m.stem.replace("model_", "").replace("000", "k") for m in selected]
        ax.set_xticklabels(step_labels, fontsize=8)
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Action %")
        ax.set_title("Action Distribution Evolution")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis="y")
        
    except Exception as e:
        ax.set_title(f"Action Evolution\n(Error: {e})")


def plot_comparison_table(log_dir: Path, ax):
    """Plot 5: Summary comparison table (V17 vs V17b metrics)."""
    # Read loss CSVs to get final metrics
    metrics = {}
    for csv_name in ["loss.csv", "td_loss.csv", "conservative_loss.csv"]:
        csv_path = log_dir / csv_name
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            values = df.iloc[:, 1].values
            name = csv_name.replace(".csv", "")
            metrics[name] = {
                "initial": f"{values[0]:.4f}",
                "final": f"{values[-1]:.4f}",
                "min": f"{values.min():.4f}",
                "reduction": f"{(1 - values[-1]/values[0])*100:.1f}%"
            }
    
    ax.axis("off")
    if not metrics:
        ax.set_title("Summary (NO DATA)")
        return
    
    headers = ["Metric", "Initial", "Final", "Min", "Reduction"]
    rows = []
    for name, m in metrics.items():
        rows.append([name, m["initial"], m["final"], m["min"], m["reduction"]])
    
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    
    # Style header
    for j, header in enumerate(headers):
        table[0, j].set_facecolor("#34495e")
        table[0, j].set_text_props(color="white", fontweight="bold")
    
    # Style rows
    for i in range(len(rows)):
        for j in range(len(headers)):
            table[i+1, j].set_facecolor("#ecf0f1" if i % 2 == 0 else "white")
    
    ax.set_title("Training Summary", fontsize=14, fontweight="bold", pad=20)


def main():
    args = parse_args()
    log_dir = args.log_dir
    
    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}")
        return 1
    
    output_dir = args.output_dir or log_dir.parent.parent / "Final_Project" / "visualize"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_name = log_dir.name
    print(f"[+] Generating diagnostics for: {run_name}")
    
    # ===== Figure 1: Loss Curves (3 subplots) =====
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
    fig1.suptitle(f"Training Loss Curves — {run_name}", fontsize=14, fontweight="bold")
    plot_loss_curves(log_dir, axes1[0], axes1[1], axes1[2])
    fig1.tight_layout(rect=[0, 0, 1, 0.93])
    path1 = output_dir / "05_training_loss_curves.png"
    fig1.savefig(path1)
    print(f"[✓] Saved {path1}")
    plt.close(fig1)
    
    # ===== Figure 2: Gradient Norms =====
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 5))
    plot_gradient_norms(log_dir, ax2)
    fig2.suptitle(f"Gradient Health — {run_name}", fontsize=14, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    path2 = output_dir / "06_gradient_norms.png"
    fig2.savefig(path2)
    print(f"[✓] Saved {path2}")
    plt.close(fig2)
    
    # ===== Figure 3: Q-Value + Action Evolution =====
    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))
    fig3.suptitle(f"Policy Analysis — {run_name}", fontsize=14, fontweight="bold")
    plot_q_value_analysis(log_dir, args.data, axes3[0])
    plot_action_distribution_evolution(log_dir, args.data, axes3[1])
    fig3.tight_layout(rect=[0, 0, 1, 0.93])
    path3 = output_dir / "07_policy_analysis.png"
    fig3.savefig(path3)
    print(f"[✓] Saved {path3}")
    plt.close(fig3)
    
    # ===== Figure 4: Summary Table =====
    fig4, ax4 = plt.subplots(1, 1, figsize=(10, 4))
    plot_comparison_table(log_dir, ax4)
    fig4.tight_layout()
    path4 = output_dir / "08_training_summary.png"
    fig4.savefig(path4)
    print(f"[✓] Saved {path4}")
    plt.close(fig4)
    
    print(f"\n[✓] All diagnostics saved to: {output_dir}")
    print(f"    05_training_loss_curves.png  — Convergence proof")
    print(f"    06_gradient_norms.png        — Stability proof")
    print(f"    07_policy_analysis.png       — Q-values + Action evolution")
    print(f"    08_training_summary.png      — Metrics table")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
