"""
Reward Signal Audit — Chứng minh khoa học rằng thay đổi Reward là hiệu quả.

Tiêu chí PASS/FAIL:
  1. [Direction]  Reward(A4) > Reward(A0) khi Queue cao  (Logic đúng hướng)
  2. [Signal]     |A4 - A0| / Range(reward) > 1%          (Tín hiệu đủ mạnh để NN phân biệt)
  3. [Queue]      Queue_max < 50,000                      (Queue nằm trong vùng vật lý hợp lệ)
"""
import pandas as pd
import numpy as np
import argparse
import sys


def audit_reward_integrity(parquet_path, label=""):
    print(f"\n{'='*60}")
    print(f" KIỂM TOÁN REWARD: {label or parquet_path}")
    print(f"{'='*60}")
    
    df = pd.read_parquet(parquet_path, columns=["action", "queue_size", "reward", "s_queue"])
    print(f"[+] Loaded {len(df):,} rows")

    # === TIÊU CHÍ 3: Queue Range ===
    q_max = df["s_queue"].max()
    q_mean = df["s_queue"].mean()
    q_median = df["s_queue"].median()
    print(f"\n[Queue] Mean={q_mean:,.0f}  Median={q_median:,.0f}  Max={q_max:,.0f}")
    queue_ok = q_max < 50_000
    print(f"  => {'✅ PASS' if queue_ok else '❌ FAIL'}: Queue max {'<' if queue_ok else '>='} 50,000")

    # === Lấy Top 10% Queue cao nhất ===
    q_threshold = df["s_queue"].quantile(0.90)
    if q_threshold == 0:
        q_threshold = df["s_queue"].mean()
    high_q = df[df["s_queue"] >= q_threshold]
    print(f"\n[High-Q Zone] {len(high_q):,} rows with Queue >= {q_threshold:,.0f}")

    # === TIÊU CHÍ 1: Direction ===
    r_by_action = {}
    for a in range(5):
        subset = high_q[high_q["action"] == a]
        if len(subset) > 0:
            r_by_action[a] = subset["reward"].mean()
            print(f"  Action {a}: reward_mean={r_by_action[a]:+.4f}  (n={len(subset):,})")
        else:
            print(f"  Action {a}: NO DATA")

    if 0 in r_by_action and 4 in r_by_action:
        diff = r_by_action[4] - r_by_action[0]
        direction_ok = diff > 0
        print(f"\n  Chênh lệch A4 - A0 = {diff:+.4f}")
        print(f"  => {'✅ PASS' if direction_ok else '❌ FAIL'}: BÁN {'tốt hơn' if direction_ok else 'TỆ HƠN'} CHỜ")
    else:
        direction_ok = False
        diff = 0
        print(f"\n  => ❌ FAIL: Không đủ dữ liệu để so sánh")

    # === TIÊU CHÍ 2: Signal Strength ===
    r_range = df["reward"].max() - df["reward"].min()
    r_std = df["reward"].std()
    if r_range > 0:
        signal_pct = abs(diff) / r_range * 100
        signal_vs_std = abs(diff) / r_std * 100
    else:
        signal_pct = 0
        signal_vs_std = 0

    signal_ok = signal_pct > 1.0
    print(f"\n[Signal Strength]")
    print(f"  Reward range      : [{df['reward'].min():.2f}, {df['reward'].max():.2f}]")
    print(f"  Reward std        : {r_std:.2f}")
    print(f"  |A4-A0| / Range  : {signal_pct:.4f}%")
    print(f"  |A4-A0| / Std    : {signal_vs_std:.4f}%")
    print(f"  => {'✅ PASS' if signal_ok else '❌ FAIL'}: Signal {'>' if signal_ok else '<='} 1% of Range")

    # === VERDICT ===
    all_pass = queue_ok and direction_ok and signal_ok
    print(f"\n{'='*60}")
    if all_pass:
        print(f" 🏆 VERDICT: ALL 3 TESTS PASSED — SẴN SÀNG TRAIN!")
    else:
        fails = []
        if not queue_ok: fails.append("Queue quá lớn")
        if not direction_ok: fails.append("Logic ngược (A0 > A4)")
        if not signal_ok: fails.append("Tín hiệu quá yếu")
        print(f" 💀 VERDICT: FAILED — {', '.join(fails)}")
        print(f"    CẤM KHÔNG ĐƯỢC MANG DATA NÀY ĐI TRAIN!")
    print(f"{'='*60}\n")
    
    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="Paths to parquet files")
    parser.add_argument("--labels", nargs="+", default=None)
    args = parser.parse_args()
    
    labels = args.labels or [f"Dataset {i+1}" for i in range(len(args.input))]
    
    results = {}
    for path, label in zip(args.input, labels):
        results[label] = audit_reward_integrity(path, label)
    
    if len(results) > 1:
        print("\n" + "="*60)
        print(" BẢNG SO SÁNH TỔNG HỢP")
        print("="*60)
        for label, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {label:30s}: {status}")
