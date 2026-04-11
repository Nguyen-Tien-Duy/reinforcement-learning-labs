# Simulation Protocol Specification for Offline RL Evaluation

## 1. Objective

This document defines the simulation protocol used to evaluate policies when direct execution-volume logs are unavailable.

Primary objective:

$$
\min \; \mathbb{E}[\text{Cost}] \quad \text{subject to} \quad \text{DeadlineMissRate} \leq \delta
$$

This protocol is designed to be fair, reproducible, and difficult to challenge in a research review.

## 2. Observed vs Simulated Variables

Observed from historical data:
- timestamp
- base_fee_per_gas
- gas_used
- gas_limit
- transaction_count

Policy-dependent and simulated:
- queue state $Q_t$
- executed amount $E_t$
- arrivals $A_t$ (if not logged directly)
- deadline timer $\tau_t$

Important note:
- Historical market sequence is fixed and identical for all policies.
- Only policy actions change simulated queue dynamics.

## 3. Environment Dynamics

Use a structural queue model:

$$
Q_{t+1} = \max(0, Q_t + A_t - E_t)
$$

$$
\tau_{t+1} = \max(0, \tau_t - 1)
$$

Terminal condition for an episode:

$$
\text{done}_t = \mathbf{1}[Q_{t+1}=0 \lor \tau_{t+1}=0]
$$

Deadline miss indicator per episode:

$$
m_e = \mathbf{1}[Q_{T_e} > 0]
$$

## 4. Action-to-Execution Mapping

Binary action setup:

$$
a_t \in \{0,1\}
$$

Execution rule:

$$
E_t = \min(Q_t, C \cdot \mathbf{1}[a_t=1])
$$

Where:
- $C$ is per-step execution capacity.
- The same $C$ is used for all compared policies.

## 5. Cost Definition

If true execution logs exist, use:

$$
\text{Cost}_t = \text{actual\_fee\_paid}_t
$$

If true execution logs are unavailable, use proxy cost:

$$
\text{Cost}^{proxy}_t = E_t \cdot g_t \cdot c_{unit}
$$

Where:
- $g_t$ is gas price signal.
- $c_{unit}$ is fixed gas-per-unit scale.

Episode-level objective:

$$
\text{Cost}_e = \sum_{t \in e} \text{Cost}_t
$$

## 6. Fairness Rules (Must Freeze Before Comparison)

1. Same market trajectory for all policies.
2. Same arrivals process and random seed for all policies.
3. Same queue initialization and deadline window.
4. Same cost function and constants.
5. Same train/validation/test time split.

These rules enforce a "same world, different policy" comparison.

## 7. Evaluation Metrics

Mandatory metrics:
- TotalCostPerEpisode
- DeadlineMissRate
- ExpectedReturn (matched) with Coverage
- ActionRate
- Risk metric (variance or CVaR)

Coverage:

$$
\text{Coverage} = \frac{1}{N}\sum_t \mathbf{1}[\hat a_t = a_t]
$$

Deadline miss rate:

$$
\text{DeadlineMissRate} = \frac{1}{|\mathcal{E}|}\sum_{e\in\mathcal{E}} m_e
$$

## 8. OPE and Uncertainty

For research-grade claims, report at least two OPE estimators:
- WIS
- DR or FQE

Also report 95% confidence intervals via episode-level bootstrap.

## 9. Sensitivity Analysis (Required)

Run scenario sweeps for:
- execution capacity $C$
- cost scale $c_{unit}$
- arrivals intensity
- deadline window

A policy is considered robust if ranking is stable across these scenarios.

## 10. Allowed Claims

Strong claim allowed:
- "Policy improves relative cost-performance trade-off under fixed simulation assumptions on unseen time windows."

Claim not allowed without production logs:
- exact real-world absolute money saved in deployment.

## 11. Minimal Reproducibility Package

Store for every run:
- dataset version and split boundaries
- simulator constants ($C$, $c_{unit}$, deadline window)
- reward coefficients
- policy checkpoint path
- random seed
- metric outputs and plots

This is the minimum requirement to make results reproducible and auditable.
