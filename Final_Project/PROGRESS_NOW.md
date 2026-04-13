# Project Progress Now

## Current status (quick view)

- [x] Problem objective and MDP documented
- [x] Raw parquet to transition builder implemented
- [x] Transition schema validator implemented
- [x] d3rlpy dataset conversion works
- [x] Codebase split into maintainable modules
- [x] Real logged action source integrated (currently proxy)
- [ ] Research-grade evaluation (OPE + CI) completed
- [ ] Final baseline comparison table completed

## What is working today

- Transition build command works and saves parquet.
- Strict validation passes on generated transition dataset.
- d3rlpy MDPDataset is created successfully.

## Main limitation right now

Current action labels are a proxy based on threshold logic from transaction_count.
This is valid for engineering integration, but not the final research dataset.

## Next 3 priorities

1. Replace proxy action with real logged action from operational data.
2. Freeze reward formula and coefficients for all experiments.
3. Train baseline set (BC, IQL, heuristic) on a time-based split.

## Daily run commands

Build transitions from raw data:

```bash
python ./Final_Project/code/simple-offline.py \
  --build-from-raw ./Final_Project/Data/data_2024-04-10_2026-04-10.parquet \
  --action-col transaction_count \
  --action-threshold 250 \
  --output ./Final_Project/Data/transitions_proxy.parquet
```

Validate transition parquet:

```bash
python ./Final_Project/code/simple-offline.py \
  --input ./Final_Project/Data/transitions_proxy.parquet
```

Convert to d3rlpy dataset:

```bash
python ./Final_Project/code/simple-offline.py \
  --input ./Final_Project/Data/transitions_proxy.parquet \
  --to-d3rlpy
```

## Read order (to know where you are)

1. RESEARCH_READINESS_CHECKLIST.md
2. OFFLINE_RL_OBJECTIVE.md
3. code/simple-offline.py
4. code/utils/offline_rl/README.md
5. code/utils/offline_rl/transition_builder.py
6. code/utils/offline_rl/build_state_action.py
7. code/utils/offline_rl/build_reward_episode.py
8. code/utils/offline_rl/validation.py
9. code/utils/offline_rl/d3rlpy_adapter.py

## Exit criteria for research-grade milestone

- Real logged actions used in final dataset.
- Time-based train/val/test split fixed and reproducible.
- OPE reported with confidence intervals.
- Baseline comparison table finalized.
- Per-run config file saved (YAML or JSON) for every experiment.
