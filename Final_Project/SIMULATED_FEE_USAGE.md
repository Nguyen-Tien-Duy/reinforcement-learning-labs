# Simulated Fee Usage Guide

## Purpose

This guide explains how to run fee simulation in evaluation mode and how to interpret the resulting cost metrics.

---

## 0) Config Fingerprint & Metadata Lockdown

### Cơ chế hoạt động

Mỗi dataset Parquet được "đóng dấu" một **fingerprint MD5** tính từ toàn bộ `TransitionBuildConfig`. Mục đích: ngăn train nhầm dataset với config không khớp.

```
TransitionBuildConfig (23 fields: deadline_penalty, urgency_beta, C_base, ...)
    │
    ▼  asdict() + json.dumps(sort_keys=True)  →  chuỗi JSON ổn định
    │
    ▼  MD5 hash
    │
    ├──► 🔒 Nhúng vào Parquet schema metadata khi BUILD
    │       key: b"config_fingerprint"
    │
    └──► 🔍 Tính lại khi AUDIT → so sánh với stored
```

> ⚠️ **Mọi field đều ảnh hưởng hash** — kể cả `normalize_state`, `queue_penalty`, `timestamp_col`...  
> Chỉ cần một field khác là fingerprint hoàn toàn khác.

### Fingerprint hiện tại (chuẩn)

| Config | Fingerprint |
|---|---|
| `action_col="gas_used"`, `normalize_state=True`, tất cả field khác = default | `31ae2421401eef6fd827e0ea40d01f51` |

### Cách kiểm tra fingerprint của một dataset

```bash
export PYTHONPATH=/mnt/WindowsD/Reinforcement\ Learning/labs/Final_Project/code && \
./venv/bin/python - << 'EOF'
import pyarrow.parquet as pq
import json, hashlib
from dataclasses import asdict
from utils.offline_rl.config import TransitionBuildConfig

DATASET = "Final_Project/Data/transitions_hardened_oracle_v2.parquet"
EXPECTED = "31ae2421401eef6fd827e0ea40d01f51"

# 1. Đọc fingerprint từ file
pf = pq.read_table(DATASET)
stored = pf.schema.metadata.get(b"config_fingerprint", b"NOT FOUND").decode()

# 2. Tính fingerprint từ config hiện tại
config = TransitionBuildConfig(action_col="gas_used", normalize_state=True)
current = hashlib.md5(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()

print(f"Stored  : {stored}")
print(f"Current : {current}")
print(f"Expected: {EXPECTED}")
print(f"Status  : {'✓ KHỚP' if stored == current == EXPECTED else '✗ LỆCH - DỪNG TRAIN!'}")
EOF
```

### Cách dùng audit tool có sẵn

```bash
export PYTHONPATH=/mnt/WindowsD/Reinforcement\ Learning/labs/Final_Project/code && \
./venv/bin/python ./Final_Project/code/scratch/training_audit_suite.py \
    --input Final_Project/Data/transitions_hardened_oracle_v2.parquet
```

Phải thấy `Config Lock: PASSED` mới được train.

---


## 1) Rebuild transitions with simulation fields

Rebuild transitions from raw data so that evaluation has `gas_t`, `queue_size`, and `executed_volume_proxy` columns.

Example:

```bash
python ./Final_Project/code/simple-offline.py \
  --build-from-raw ./Final_Project/Data/data_2024-04-10_2026-04-10.parquet \
  --action-col transaction_count \
  --action-threshold 250 \
  --output ./Final_Project/Data/transitions_proxy.parquet
```
```bash
# 1. Thiết lập đường dẫn
export PYTHONPATH=/mnt/WindowsD/Reinforcement\ Learning/labs/Final_Project/code

# 2. Tạo Dataset Hardened (explicit config — mọi tham số kinh tế phải khai rõ)
./venv/bin/python ./Final_Project/code/simple-offline.py \
    --build-from-raw Final_Project/Data/data_2024-04-10_2026-04-10.parquet \
    --output Final_Project/Data/transitions_hardened_oracle_v2.parquet \
    --action-col gas_used \
    --use-oracle \
    --oracle-mix-ratio 0.5 \
    --suboptimal-mix-ratio 0.2 \
    --deadline-penalty 5000000000.0 \
    --urgency-beta 100.0 \
    --urgency-alpha 3.0 \
    --reward-scale 1000000.0 \
    --C-base 21000.0 \
    --C-mar 15000.0 \
    --fee-gas-scale 1e9 \
    --execution-capacity 500.0 \
    --episode-hours 24 \
    --history-window 3 \
    --to-d3rlpy
```

>  **Quan trọng**: Khi build data, **phải khai báo tường minh** tất cả tham số kinh tế.  
> Dùng CLI default là không an toàn — nếu default thay đổi, data sẽ có fingerprint khác mà không có cảnh báo.  
> Fingerprint hiện tại của config này: `31ae2421401eef6fd827e0ea40d01f51`


evaluation: to check data intergrity in data transition 

```bash
./venv/bin/python ./Final_Project/code/scratch/training_audit_suite.py --input Final_Project/Data/transitions_hardened_v2.parquet
```

## 1a) Training command

> **Hyperparameter rationale**
> - `sample-size 1000000`: dùng 1M/5.2M transitions (20%) để đa dạng data, tránh overfit
> - `n-steps 400000` × `batch 512` / `1,000,000` ≈ **205 lần/sample** — hơi cao, nhưng IQL với dense reward chịu được
> - `expectile 0.9`: value function hướng tới top 10% Oracle actions (thay vì 0.8 = top 20%)
> - `gamma 0.99`: giữ nguyên vì urgency_penalty là dense reward, không cần giảm

```bash
export PYTHONPATH=/mnt/WindowsD/Reinforcement\ Learning/labs/Final_Project/code && \
nohup ./venv/bin/python ./Final_Project/code/simple-offline.py \
    --input Final_Project/Data/transitions_hardened_oracle.parquet \
    --train-toy \
    --n-steps 100000 \
    --sample-size 500000 \
    --save-interval 5000 \
    --seed 42 \
    --skip-validation \
    > train_v1.log 2>&1 &

echo "PID: $!"
```

> **Lưu ý**: Các tham số kinh tế (`--deadline-penalty`, `--urgency-beta`...) **không cần thiết** khi train trên dataset đã build sẵn — reward đã được mã hóa trong file parquet. Chúng chỉ cần khi `--build-from-raw` (tạo data) hoặc `--evaluate` (simulation).



## 2) Run evaluation with simulated fee

### 2a) Evaluate a single model

```bash
# Note for Fish shell users: use `set -x PYTHONPATH /mnt/WindowsD/Reinforcement\ Learning/labs/Final_Project/code` instead of export
export PYTHONPATH=/mnt/WindowsD/Reinforcement\ Learning/labs/Final_Project/code && \
./venv/bin/python ./Final_Project/code/simple-offline.py \
    --input Final_Project/Data/transitions_hardened_v2.parquet \
    --evaluate \
    --model-path d3rlpy_logs/IQL_HardPenalty_<timestamp>/<model_name>.d3 \
    --eval-ratio 0.2 \
    --limit-eval-episodes 100 \
    --deadline-penalty 5000000000.0 \
    --urgency-beta 100.0 \
    --urgency-alpha 3.0 \
    --reward-scale 1000000.0 \
    --C-base 21000.0 \
    --C-mar 15000.0 \
    --fee-gas-scale 1e9 \
    --execution-capacity 500.0 \
    --episode-hours 24 \
    --history-window 3 \
    --action-col gas_used \
    --skip-validation
```

### 2b) Evaluate all checkpoints (Leaderboard)

```bash
# Note for Fish shell users: use `set -x PYTHONPATH /mnt/WindowsD/Reinforcement\ Learning/labs/Final_Project/code` instead of export
export PYTHONPATH=/mnt/WindowsD/Reinforcement\ Learning/labs/Final_Project/code && \
./venv/bin/python ./Final_Project/code/simple-offline.py \
    --input Final_Project/Data/transitions_hardened_v2.parquet \
    --evaluate \
    --eval-all \
    --model-path d3rlpy_logs/IQL_HardPenalty_<timestamp>/ \
    --eval-ratio 0.2 \
    --limit-eval-episodes 30 \
    --deadline-penalty 5000000000.0 \
    --urgency-beta 100.0 \
    --urgency-alpha 3.0 \
    --reward-scale 1000000.0 \
    --C-base 21000.0 \
    --C-mar 15000.0 \
    --fee-gas-scale 1e9 \
    --execution-capacity 500.0 \
    --episode-hours 24 \
    --history-window 3 \
    --action-col gas_used \
    --skip-validation
```

## 3) New parameters

- `--fee-gas-scale`
  - Converts gas unit before cost aggregation.
  - Recommended default: `1e9` (Wei to Gwei).

- `--execution-proxy-mode`
  - `queue`: cost uses queue-based executed volume proxy.
  - `unit`: cost uses 1 unit per execute action.

- `--execution-capacity`
  - Optional cap on executed volume proxy per step.
  - Useful for stress and sensitivity tests.

## 4) Cost equations used in evaluation

If gas column is available, simulated step cost is:

$$
\text{Cost}^{sim}_t = \frac{g_t}{s_g} \cdot E^{proxy}_t
$$

Where:
- $g_t$: gas value from data (`gas_t` or fallback gas column).
- $s_g$: `fee-gas-scale`.
- $E^{proxy}_t$: execution proxy from selected mode.

Episode cost:

$$
\text{Cost}^{sim}_e = \sum_{t \in e} \text{Cost}^{sim}_t
$$

Reported metric:

$$
\text{TotalCostPerEpisode} = \frac{1}{|\mathcal{E}|}\sum_e \text{Cost}^{sim}_e
$$

## 5) Interpretation

- `total_cost_per_episode`: primary simulated objective metric.
- `total_cost_sum`: total simulated cost over holdout episodes.
- `execution_proxy_per_episode`: average simulated executed volume per episode.

## 6) Reporting discipline

When publishing results:
- Explicitly state this is simulated/proxy cost, not production paid fee.
- Keep `fee-gas-scale`, proxy mode, and capacity fixed across policy comparisons.
- Report DeadlineMissRate together with cost (constraint + objective view).
