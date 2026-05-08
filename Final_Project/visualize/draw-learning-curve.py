from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent

log_dir = BASE_DIR / "d3rlpy_logs" / "DiscreteCQL_20260411185238"

# Load the log data
def load_metric_csv(path: Path) -> pd.DataFrame:
    # d3rlpy csv format: epochs, step, value (not include header)
    df = pd.read_csv(path, header=None, names=['epochs', 'step', 'value'])
    df = df.sort_values('step').reset_index(drop=True)
    return df

loss_df = load_metric_csv(log_dir / "loss.csv")
td_df = load_metric_csv(log_dir / "td_loss.csv")
cql_df = load_metric_csv(log_dir / "conservative_loss.csv")

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

axes[0].plot(loss_df["step"], loss_df["value"], marker="o", linewidth=2)
axes[0].set_title("Training Loss")
axes[0].set_ylabel("loss")
axes[0].set_yscale("log")
axes[0].grid(True, alpha=0.3)

axes[1].plot(td_df["step"], td_df["value"], marker="o", linewidth=2, color="tab:orange")
axes[1].set_title("TD Loss")
axes[1].set_ylabel("td_loss")
axes[1].set_yscale("log")
axes[1].grid(True, alpha=0.3)

axes[2].plot(cql_df["step"], cql_df["value"], marker="o", linewidth=2, color="tab:green")
axes[2].set_title("Conservative Loss")
axes[2].set_ylabel("conservative_loss")
axes[2].set_xlabel("training step")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(log_dir / "learning_curve.png", dpi=180)
plt.show()