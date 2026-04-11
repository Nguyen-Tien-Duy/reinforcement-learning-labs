# Simulated Fee Usage Guide

## Purpose

This guide explains how to run fee simulation in evaluation mode and how to interpret the resulting cost metrics.

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

## 2) Run evaluation with simulated fee

Example command:

```bash
python ./Final_Project/code/simple-offline.py \
  --input ./Final_Project/Data/transitions_proxy.parquet \
  --evaluate \
  --model-path ./Final_Project/output/toy_iql_YYYYMMDD_HHMMSS.d3 \
  --eval-ratio 0.2 \
  --fee-gas-scale 1000000000 \
  --execution-proxy-mode queue
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
