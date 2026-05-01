"""
Plot Training Curves — Discrete CQL V6
Vẽ TD Loss và Conservative Loss từ CSV logs của d3rlpy.

Usage:
    python Final_Project/visualize/plot_training_curves.py \
        --log_dir d3rlpy_logs/DiscreteCQL_V6_20260428_0426_20260428042630 \
        --output report/Images/05_training_curves.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────────
BG = "#0F1117"; PANEL = "#1A1D27"; TEXT = "#E2E8F0"
BLUE = "#4C9BE8"; ORANGE = "#F4956A"; PURPLE = "#A78BFA"; GREEN = "#50C878"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": "#2D3148", "axes.labelcolor": TEXT,
    "text.color": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    "grid.color": "#2A2D3E", "grid.alpha": 0.5,
    "font.family": "DejaVu Sans", "font.size": 10,
})


def read_csv(path):
    """Read d3rlpy metric CSV: epoch, step, value."""
    steps, values = [], []
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            steps.append(int(row[1]))
            values.append(float(row[2]))
    return np.array(steps), np.array(values)


def main():
    parser = argparse.ArgumentParser(description="Plot CQL training curves")
    parser.add_argument("--log_dir",
                        default="d3rlpy_logs/DiscreteCQL_V6_20260428_0426_20260428042630")
    parser.add_argument("--output", default="report/Images/05_training_curves.png")
    parser.add_argument("--sota_step", type=int, default=160000,
                        help="Checkpoint step to highlight as SOTA")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    steps_td, td_loss = read_csv(log_dir / "loss.csv")
    steps_cons, cons_loss = read_csv(log_dir / "conservative_loss.csv")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Discrete CQL V6 — Training Curves (500K steps)",
                 fontsize=14, color=TEXT, fontweight="bold")

    # TD Loss
    ax1.plot(steps_td / 1000, td_loss, color=ORANGE, lw=2, marker="o", ms=3,
             label="TD Loss")
    ax1.axvline(args.sota_step / 1000, color=GREEN, lw=1.5, ls="--", alpha=0.7,
                label=f"Model {args.sota_step // 1000}K (SOTA)")
    ax1.set_ylabel("TD Loss (Bellman Error)", fontsize=11)
    ax1.legend(fontsize=9, framealpha=0.3)
    ax1.grid(True, lw=0.4)
    ax1.set_title("TD Loss — Q-value Magnitude Growth", fontsize=11, color=ORANGE)

    # Conservative Loss
    ax2.plot(steps_cons / 1000, cons_loss, color=PURPLE, lw=2, marker="s", ms=3,
             label="Conservative Loss")
    ax2.axvline(args.sota_step / 1000, color=GREEN, lw=1.5, ls="--", alpha=0.7,
                label=f"Model {args.sota_step // 1000}K (SOTA)")
    ax2.set_ylabel("Conservative Loss", fontsize=11)
    ax2.set_xlabel("Training Steps (×1000)", fontsize=11)
    ax2.legend(fontsize=9, framealpha=0.3)
    ax2.grid(True, lw=0.4)
    ax2.set_title("Conservative Loss — In-Distribution vs OOD Separation",
                  fontsize=11, color=PURPLE)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[✓] Saved {args.output}")


if __name__ == "__main__":
    main()
