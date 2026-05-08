"""
State Feature Monitor — Discrete V6/V8
Phân tích phân phối và tương quan 11 chiều state vector trong transition dataset.
Output: ảnh PNG lưu vào cùng thư mục với script này.

Usage:
    python Final_Project/visualize/state_monitor.py \
        --input Final_Project/Data/transitions_discrete_v6.parquet \
        [--sample 200000]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
import pyarrow.parquet as pq

# ── State column resolution ──────────────────────────────────────────────────
_COL_CANDIDATES = {
    "s_gas_t0": ["s_gas_t0", "s_gas_0"],
    "s_gas_t1": ["s_gas_t1", "s_gas_1"],
    "s_gas_t2": ["s_gas_t2", "s_gas_2"],
    "s_congestion":   ["s_congestion"],
    "s_momentum":     ["s_momentum"],
    "s_acceleration": ["s_acceleration", "s_accel"],
    "s_surprise":     ["s_surprise"],
    "s_backlog":      ["s_backlog"],
    "s_queue":        ["s_queue"],
    "s_time":         ["s_time", "s_time_left"],
    "s_gas_ref":      ["s_gas_ref"],
}

LABELS = [
    "Gas t (raw)", "Gas t-1 (raw)", "Gas t-2 (raw)",
    "Congestion p_t", "Momentum m_t", "Acceleration a_t",
    "Surprise u_t", "Backlog b_t",
    "Queue size Q_t", "Time to deadline τ_t", "Gas ref (rolling)",
]

def resolve_state_cols(available_cols):
    """Map canonical names → actual column names found in parquet."""
    resolved = []
    mapping = {}
    for canonical, candidates in _COL_CANDIDATES.items():
        found = next((c for c in candidates if c in available_cols), None)
        if found:
            resolved.append(found)
            mapping[canonical] = found
        else:
            print(f"[!] Warning: Missing state col: {canonical} (tried {candidates})")
            resolved.append(None)
    return resolved, mapping

# ── Palette ───────────────────────────────────────────────────────────────────
BLUE   = "#4C9BE8"
ORANGE = "#F4956A"
PURPLE = "#A78BFA"
BG     = "#0F1117"
PANEL  = "#1A1D27"
TEXT   = "#E2E8F0"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   PANEL,
    "axes.edgecolor":   "#2D3148",
    "axes.labelcolor":  TEXT,
    "text.color":       TEXT,
    "xtick.color":      TEXT,
    "ytick.color":      TEXT,
    "grid.color":       "#2A2D3E",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
    "font.size":        9,
})


def load_data(path: str, sample: int, state_cols: list) -> pd.DataFrame:
    print(f"[+] Loading {path} ...", flush=True)
    # Filter out None from state_cols
    load_cols = [c for c in state_cols if c] + ["action", "episode_id"]
    df = pd.read_parquet(path, columns=load_cols)
    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=42)
    print(f"[+] Loaded {len(df):,} rows × {len(df.columns)} cols", flush=True)
    return df


# ── Plot 1: Distribution grid (11 histograms) ─────────────────────────────────
def plot_distributions(df: pd.DataFrame, out_dir: Path, state_cols: list):
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle("State Feature Distributions — Transition Dataset", fontsize=14,
                 color=TEXT, fontweight="bold", y=0.98)
    axes_flat = axes.flatten()

    for i, (col, label) in enumerate(zip(state_cols, LABELS)):
        ax = axes_flat[i]
        if col is None or col not in df.columns:
            ax.text(0.5, 0.5, "Column Missing", ha='center', va='center')
            continue
            
        vals = df[col].dropna().values
        if len(vals) == 0: continue

        # clip extreme outliers for readability (keep 99.5%)
        lo, hi = np.percentile(vals, 0.25), np.percentile(vals, 99.75)
        vals_c = np.clip(vals, lo, hi)

        ax.hist(vals_c, bins=80, color=BLUE, alpha=0.85, edgecolor="none")
        ax.set_title(f"[{i}] {label}", fontsize=8.5, color=PURPLE, pad=4)
        ax.set_xlabel("Value", fontsize=7.5)
        ax.set_ylabel("Count", fontsize=7.5)
        ax.grid(True, linewidth=0.4)

        # stats annotation
        mu, md, sd = vals.mean(), np.median(vals), vals.std()
        ax.axvline(mu, color=ORANGE, lw=1.2, linestyle="--", label=f"μ={mu:.3g}")
        ax.axvline(md, color="white", lw=0.8, linestyle=":", label=f"m={md:.3g}")
        ax.legend(fontsize=6.5, loc="upper right",
                  labelcolor=TEXT, framealpha=0.3)

    # hide unused panels
    for j in range(len(state_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)
        
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = out_dir / "01_state_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[✓] Saved {out}")


# ── Plot 2: Correlation heatmap ───────────────────────────────────────────────
def plot_correlation(df: pd.DataFrame, out_dir: Path, state_cols: list):
    cols = [c for c in state_cols if c and c in df.columns]
    state_df = df[cols].dropna()
    corr = state_df.corr()

    fig, ax = plt.subplots(figsize=(13, 10))
    fig.suptitle("State Feature Correlation Matrix", fontsize=13,
                 color=TEXT, fontweight="bold")

    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        corr, ax=ax, cmap=cmap, center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 7.5},
        linewidths=0.4, linecolor="#2A2D3E",
        xticklabels=[c.replace('s_','') for c in cols],
        yticklabels=[c.replace('s_','') for c in cols],
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
    )
    ax.set_title("Pearson Correlation — State Features", color=TEXT, pad=8)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    fig.tight_layout()
    out = out_dir / "02_state_correlation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[✓] Saved {out}")


# ── Plot 3: Action distribution + state-by-action boxplots ───────────────────
def plot_action_analysis(df: pd.DataFrame, out_dir: Path, mapping: dict):
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Action Distribution & State-by-Action Breakdown", fontsize=13,
                 color=TEXT, fontweight="bold")

    # Row 0: action bar chart (full width)
    ax0 = fig.add_subplot(gs[0, :])
    counts = df["action"].value_counts(normalize=True).sort_index() * 100
    bars = ax0.bar(counts.index, counts.values,
                   color=[BLUE, PURPLE, ORANGE, "#50C878", "#FF6B6B"],
                   width=0.6, edgecolor="none")
    for bar, v in zip(bars, counts.values):
        ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=9, color=TEXT)
    ax0.set_xticks([0, 1, 2, 3, 4])
    ax0.set_xticklabels(["0 (0%)", "1 (25%)", "2 (50%)", "3 (75%)", "4 (100%)"])
    ax0.set_xlabel("Action bin → execution ratio")
    ax0.set_ylabel("% of dataset")
    ax0.set_title("Action Distribution  ← KEY: should be balanced for CQL to learn",
                  color=ORANGE, fontsize=10)
    ax0.grid(True, axis="y", linewidth=0.4)

    # Rows 1-3: box plots of selected state features per action
    key_features = [
        (mapping.get("s_queue"),       "Queue Q_t",       8),
        (mapping.get("s_time"),        "Time to DL τ_t",  9),
        (mapping.get("s_congestion"),  "Congestion p_t",  3),
        (mapping.get("s_momentum"),    "Momentum m_t",    4),
        (mapping.get("s_backlog"),     "Backlog b_t",     7),
        (mapping.get("s_gas_t0"),      "Gas price",       0),
        (mapping.get("s_surprise"),    "Surprise u_t",    6),
        (mapping.get("s_gas_ref"),     "Gas ref",        10),
        (mapping.get("s_acceleration"),"Accel a_t",       5),
    ]

    palette = {0: BLUE, 1: PURPLE, 2: ORANGE, 3: "#50C878", 4: "#FF6B6B"}

    for idx, (col, title, feat_idx) in enumerate(key_features):
        if col is None or col not in df.columns: continue
        row = (idx // 3) + 1
        col_idx = idx % 3
        ax = fig.add_subplot(gs[row, col_idx])

        data = [df[df["action"] == a][col].dropna().values for a in range(5)]
        # clip outliers
        global_99 = np.percentile(df[col].dropna(), 99)
        data_c = [np.clip(d, -np.inf, global_99) for d in data]

        bp = ax.boxplot(data_c, patch_artist=True, medianprops={"color": "white", "lw": 1.2},
                        whiskerprops={"color": "#8888AA"}, capprops={"color": "#8888AA"},
                        flierprops={"marker": ".", "ms": 2, "alpha": 0.3, "color": "#8888AA"})
        for patch, (act, color) in zip(bp["boxes"], palette.items()):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(f"[{feat_idx}] {title}", fontsize=8.5, color=PURPLE, pad=3)
        ax.set_xticklabels([f"a={i}" for i in range(5)], fontsize=7.5)
        ax.set_xlabel("Action bin", fontsize=7)
        ax.grid(True, axis="y", linewidth=0.3)

    out = out_dir / "03_action_state_analysis.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[✓] Saved {out}")


# ── Plot 4: Time-series snapshot (1 episode) ──────────────────────────────────
def plot_episode_snapshot(df: pd.DataFrame, out_dir: Path, mapping: dict):
    q_col = mapping.get("s_queue")
    t_col = mapping.get("s_time")
    gas_col = mapping.get("s_gas_t0")
    cong_col = mapping.get("s_congestion")
    back_col = mapping.get("s_backlog")
    
    if not all([q_col, t_col, gas_col]): 
        print("[!] Essential columns for snapshot missing, skipping Plot 4.")
        return

    # Pick the episode with median queue size
    ep_group = df.groupby("episode_id")[q_col].median()
    target_ep = (ep_group - ep_group.median()).abs().idxmin()
    ep_df = df[df["episode_id"] == target_ep].sort_values(t_col, ascending=False)

    if len(ep_df) < 10:
        print("[!] Episode too short for snapshot, skipping.")
        return

    t = np.arange(len(ep_df))

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f"Episode Snapshot (ep_id={target_ep}, n={len(ep_df)} steps)",
                 fontsize=12, color=TEXT, fontweight="bold")

    # Gas price
    axes[0].plot(t, ep_df[gas_col].values, color=ORANGE, lw=0.7)
    axes[0].set_ylabel("Gas price\n(raw wei)", fontsize=8)
    axes[0].grid(True, linewidth=0.3)
    axes[0].set_title("Gas Price over Episode", fontsize=9, color=ORANGE)

    # Queue size
    axes[1].fill_between(t, ep_df[q_col].values, color=BLUE, alpha=0.6)
    axes[1].set_ylabel("Queue Q_t", fontsize=8)
    axes[1].grid(True, linewidth=0.3)
    axes[1].set_title("Queue Size (should reach 0 before deadline)", fontsize=9, color=BLUE)

    # Action
    action_colors = [["#4C9BE8","#A78BFA","#F4956A","#50C878","#FF6B6B"][int(a)]
                     for a in ep_df["action"].values]
    axes[2].bar(t, ep_df["action"].values, color=action_colors, width=1.0, edgecolor="none")
    axes[2].set_yticks([0, 1, 2, 3, 4])
    axes[2].set_ylabel("Action bin", fontsize=8)
    axes[2].set_title("Action chosen by Oracle", fontsize=9, color=PURPLE)
    axes[2].grid(True, axis="y", linewidth=0.3)

    # Congestion + Backlog
    if cong_col and cong_col in ep_df.columns:
        axes[3].plot(t, ep_df[cong_col].values, color=ORANGE, lw=0.7, label="Congestion p_t")
    if back_col and back_col in ep_df.columns:
        axes[3].plot(t, ep_df[back_col].values, color=PURPLE, lw=0.7, label="Backlog b_t")
    axes[3].axhline(0, color="#555577", lw=0.5)
    axes[3].set_ylabel("Congestion / Backlog", fontsize=8)
    axes[3].set_xlabel("Block step within episode", fontsize=8)
    axes[3].legend(fontsize=7.5, loc="upper right", framealpha=0.3)
    axes[3].grid(True, linewidth=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = out_dir / "04_episode_snapshot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[✓] Saved {out}")


# ── Plot 5: Summary statistics table ──────────────────────────────────────────
def print_stats(df: pd.DataFrame, state_cols: list):
    cols = [c for c in state_cols if c and c in df.columns]
    print("\n" + "=" * 70)
    print("STATE FEATURE SUMMARY STATISTICS")
    print("=" * 70)
    stats = df[cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    with pd.option_context("display.float_format", "{:.4g}".format,
                           "display.max_columns", 20, "display.width", 120):
        print(stats.T.to_string())
    print()
    print("ACTION DISTRIBUTION:")
    dist = df["action"].value_counts(normalize=True).sort_index() * 100
    for k, v in dist.items():
        bar = "█" * int(v / 2)
        target = " ← target: < 50% for a=0" if k == 0 else ""
        print(f"  action={k} ({['0%','25%','50%','75%','100%'][k]:>4s}): {v:5.1f}%  {bar}{target}")
    print()
    # Correlation with action (Spearman, more robust)
    print("SPEARMAN CORRELATION with ACTION:")
    corr = df[cols + ["action"]].corr(method="spearman")["action"].drop("action")
    for col, r in corr.sort_values(key=abs, ascending=False).items():
        bar = "█" * int(abs(r) * 20)
        sign = "+" if r >= 0 else "-"
        print(f"  {col:20s}: {sign}{abs(r):.3f}  {bar}")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="State feature monitor for transition dataset")
    parser.add_argument("--input",  default="Final_Project/Data/transitions_discrete_v6.parquet")
    parser.add_argument("--sample", type=int, default=300_000,
                        help="Number of rows to sample (0 = all)")
    args = parser.parse_args()

    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Peek at schema to resolve column names
    try:
        schema = pq.read_schema(args.input)
        available_cols = schema.names
    except Exception as e:
        print(f"[!] Error reading schema from {args.input}: {e}")
        sys.exit(1)

    state_cols, mapping = resolve_state_cols(available_cols)
    
    df = load_data(args.input, args.sample, state_cols)

    print_stats(df, state_cols)

    print("\n[+] Generating plots...")
    plot_distributions(df, out_dir, state_cols)
    plot_correlation(df, out_dir, state_cols)
    plot_action_analysis(df, out_dir, mapping)
    plot_episode_snapshot(df, out_dir, mapping)

    print(f"\n[✓] All plots saved to: {out_dir.resolve()}/")
    print("    01_state_distributions.png  — Histogram 11 features")
    print("    02_state_correlation.png    — Pearson correlation heatmap")
    print("    03_action_state_analysis.png — Action distribution + boxplots")
    print("    04_episode_snapshot.png     — Time-series of 1 episode")


if __name__ == "__main__":
    main()
