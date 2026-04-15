# Simulated Fee Usage Guide (V20 - Solvable Universe)

## Purpose

This guide explains how to **Build**, **Train**, **Evaluate**, and **Audit** the Offline RL gas fee optimization agent.

> **⚠️ CRITICAL:** All 3 stages (Build, Train, Evaluate) **MUST** use the same physics parameters.
> Mixing parameters from different versions (e.g., V17 params with V20 data) will produce 100% miss rate.

---

## Quick Reference: V20 Physics Parameters (Solvable Universe)

| Parameter | Value | Rationale |
|:---|:---|:---|
| `--execution-capacity` | `500.0` | High throughput to ensure system is never bottlenecked. |
| `--arrival-scale` | `0.05` | **The Key Fix:** Scale arrivals to ~70% load so the problem is mathematically solvable. |
| `--urgency-beta` | `50.0` | High linear signal for pending items. |
| `--urgency-alpha` | `3.0` | Exponential urgency growth near deadline. |
| `--deadline-penalty` | `1,000,000.0` | Extreme penalty to force Agent to clear queue at all costs. |
| `--reward-scale` | `1.0` | Use Robust Scaling (Median/IQR) instead of fixed scaling. |
| `--oracle-mix-ratio` | `0.9` | 90% Oracle actions (learn the right thing). |
| `--suboptimal-mix-ratio` | `0.05` | 5% inverted actions (learn crisis handling). |
| `--cql-alpha` | `0.1` | Low conservatism so AI dares to pick Action 3-4. |

---

## 1) Build Dataset (V20)

Build the training dataset with **Spec-aligned Reward**, **Robust Arrival Scaling**, and **High-Precision Oracle**.

**Design philosophy:**
- `arrival_scale=0.05`: Ensures average demand is well below capacity.
- `deadline_penalty=1,000,000`: Death-sentence for missing a deadline.
- `PYTHONDONTWRITEBYTECODE=1`: Crucial to avoid stale Numba/Python cache.

```fish
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 PYTHONPATH="/mnt/WindowsD/Reinforcement Learning/labs/Final_Project/code" \
nohup ./venv/bin/python ./Final_Project/code/simple-offline.py \
    --build-from-raw Final_Project/Data/data_2024-04-10_2026-04-10.parquet \
    --output Final_Project/Data/transitions_discrete_v20.parquet \
    --action-col gas_used \
    --use-oracle \
    --oracle-mix-ratio 0.9 \
    --suboptimal-mix-ratio 0.05 \
    --deadline-penalty 1000000.0 \
    --urgency-beta 50.0 \
    --urgency-alpha 3.0 \
    --reward-scale 1.0 \
    --execution-capacity 500.0 \
    --arrival-scale 0.05 \
    --episode-hours 24 \
    --history-window 3 \
    --skip-validation \
    > build_v20.log 2>&1 &

tail -f build_v20.log
```

**Expected output:**
- `Queue range: [0, ~130,000]` — Mean ~654, Median ~204
- `Action unique values: [0, 1, 2, 3, 4]`

---

## 2) Audit Dataset (MANDATORY before training)

Run the 3-criteria audit to verify data quality:

```fish
PYTHONPATH="/mnt/WindowsD/Reinforcement Learning/labs/Final_Project/code" \
./venv/bin/python ./Final_Project/visualize/check_oracle_miss.py \
    Final_Project/Data/transitions_discrete_v20.parquet
```

**Must pass:**
- ✅ **Direction:** `A4 - A0 > 0` (executing is better than waiting)
- ✅ **Signal Strength:** `|A4-A0| / Range > 1%` (NN can distinguish actions)
- ⚠️ **Queue Max:** May fail due to spike episodes — acceptable if Mean Queue < 1,000

---

## 3) Visualize State Distribution

Check action-state correlations before training:

```fish
./venv/bin/python Final_Project/visualize/state_monitor.py \
    --input Final_Project/Data/transitions_discrete_v20.parquet
```

**Key metrics to verify:**
- `s_queue` correlation with action: **must be POSITIVE** (e.g., +0.338)
- `action=0` distribution: **must be < 50%** (e.g., 32.9%)
- If `s_queue` correlation is negative → data is broken, DO NOT TRAIN

---

## 4) Train (DiscreteCQL V20)

```fish
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 PYTHONPATH="/mnt/WindowsD/Reinforcement Learning/labs/Final_Project/code" \
nohup ./venv/bin/python ./Final_Project/code/simple-offline.py \
    --input Final_Project/Data/transitions_discrete_v20.parquet \
    --train-toy \
    --n-steps 100000 \
    --sample-size 5224088 \
    --save-interval 5000 \
    --reward-scale 1.0 \
    --deadline-penalty 1000000.0 \
    --urgency-beta 50.0 \
    --arrival-scale 0.05 \
    --execution-capacity 500.0 \
    --cql-alpha 0.1 \
    --skip-validation \
    > train_v20.log 2>&1 &

tail -f train_v20.log
```

> **Hyperparameter rationale:**
> - `cql-alpha=0.1`: Critical! Default 1.0 is too conservative — AI only picks Action 1 (25%). At 0.1, AI dares to pick Action 3-4 when needed.
> - `sample-size=5224088`: Use full dataset (no subsampling).
> - `n-steps=100000`: 20 epochs × 5000 steps/epoch.

**Expected training metrics:**
- `loss` ≈ 1.5-2.0 (stable, not diverging)
- `td_loss` ≈ 0.5-0.8
- `conservative_loss` ≈ 1.0

---

## 5) Evaluate — Single Model

```fish
env OMP_NUM_THREADS=3 PYTHONPATH="/mnt/WindowsD/Reinforcement Learning/labs/Final_Project/code" \
./venv/bin/python ./Final_Project/code/simple-offline.py \
    --input Final_Project/Data/transitions_discrete_v17.parquet \
    --evaluate \
    --model-path d3rlpy_logs/DiscreteCQL_V6_<timestamp>/model_100000.d3 \
    --eval-ratio 0.2 \
    --limit-eval-episodes 20 \
    --deadline-penalty 10000.0 \
    --urgency-beta 10.0 \
    --urgency-alpha 3.0 \
    --reward-scale 1.0 \
    --execution-capacity 200.0 \
    --arrival-scale 0.5 \
    --episode-hours 24 \
    --history-window 3 \
    --action-col gas_used \
    --skip-validation
```

---

## 6) Evaluate — All Checkpoints (Leaderboard)

```fish
env OMP_NUM_THREADS=3 PYTHONPATH="/mnt/WindowsD/Reinforcement Learning/labs/Final_Project/code" \
nohup ./venv/bin/python ./Final_Project/code/simple-offline.py \
    --input Final_Project/Data/transitions_discrete_v17.parquet \
    --evaluate \
    --eval-all \
    --model-path d3rlpy_logs/DiscreteCQL_V6_<timestamp>/ \
    --eval-ratio 0.2 \
    --limit-eval-episodes 20 \
    --deadline-penalty 10000.0 \
    --urgency-beta 10.0 \
    --urgency-alpha 3.0 \
    --reward-scale 1.0 \
    --execution-capacity 200.0 \
    --arrival-scale 0.5 \
    --episode-hours 24 \
    --history-window 3 \
    --action-col gas_used \
    --skip-validation \
    > evaluate_v17.log 2>&1 &

tail -f evaluate_v17.log
```

---

## 7) Evaluate — Live Watch (Auto-evaluate new checkpoints)

```fish
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="/mnt/WindowsD/Reinforcement Learning/labs/Final_Project/code" \
./venv/bin/python ./Final_Project/code/simple-offline.py \
    --input Final_Project/Data/transitions_discrete_v20.parquet \
    --evaluate \
    --eval-watch \
    --model-path d3rlpy_logs/DiscreteCQL_V20_<timestamp>/ \
    --eval-ratio 0.2 \
    --limit-eval-episodes 10 \
    --deadline-penalty 1000000.0 \
    --urgency-beta 50.0 \
    --urgency-alpha 3.0 \
    --reward-scale 1.0 \
    --execution-capacity 500.0 \
    --arrival-scale 0.05 \
    --episode-hours 24 \
    --history-window 3 \
    --action-col gas_used \
    --skip-validation
```

> **Note:** `--eval-watch` loops continuously. Use Ctrl+C to stop.

---

## 8) Reward Design (V17 Spec)

The reward follows `REWARD_DESIGN_SPEC.md` with 3 components:

### 8.1) Efficiency
$$R_{eff} = n_t \times \frac{g_{ref} - g_t}{s_g}, \quad s_g = 10$$

- No `C_base` overhead (removed in V12+).
- Positive when executing at below-average gas price.

### 8.2) Urgency
$$R_{urg} = \beta \cdot q_t \cdot e^{\alpha(1 - \tau_t / D)}$$

- **No Clipping:** In V20, we use pure linear scaling to preserve high-fidelity signals.
- Range is managed via Robust Normalization (Median/IQR) during training preprocessing.

### 8.3) Catastrophe
$$R_{cat} = -\lambda_d \cdot \mathbf{1}[q_T > 0]$$

- Applied only at episode end.

### Total reward per step:
$$R_t = R_{eff} - R_{urg} - R_{cat}$$

---

## 9) Known Issues & Gotchas

| Issue | Symptom | Fix |
|:---|:---|:---|
| Wrong eval params | Miss rate = 100%, cost = 700k+ | Use exact V17 params from this guide |
| `R_overhead` NameError | Eval crashes | Already fixed in `enviroment.py` |
| `eval_df["state"]` KeyError | Eval crashes | Already fixed (uses `STATE_COLS`) |
| CQL alpha too high | AI only picks Action 1 (25%) | Use `--cql-alpha 0.1` |
| Missing `arrival_scale` in env | Queue explodes during eval | Already fixed in `enviroment.py` |
| `--eval-watch` loops forever | Process never exits | Use Ctrl+C or `--eval-all` instead |
| Numba cache stale | Oracle uses old formula | `find . -name "__pycache__" -exec rm -rf {} +` |

---

## 10) Validation Checklist

Before submitting results, verify ALL of these:

- [ ] Audit passes Direction test (A4 > A0)
- [ ] Audit passes Signal test (> 1% of range)
- [ ] State monitor shows `s_queue` correlation > +0.1
- [ ] State monitor shows `action=0` < 50%
- [ ] Training loss is stable (not diverging)
- [ ] Evaluation uses **exact same** physics params as build
- [ ] Miss rate < 1.0 on at least some checkpoints
