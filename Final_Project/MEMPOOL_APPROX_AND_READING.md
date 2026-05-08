# Mempool Approximation Notes 

## Goal

This note explains how to proceed when mempool data is unavailable, using only block-level fields already present in this project:
- block_number
- timestamp
- base_fee_per_gas
- gas_used
- gas_limit
- transaction_count
- size

## Key idea

No mempool does not mean no RL.
It means the problem is partially observed, so we estimate hidden demand pressure using proxies.

## Practical pseudo-mempool features

### 1) Block congestion pressure

Use block fullness against EIP-1559 target gas:

$$
\text{target\_gas}_t = \frac{\text{gas\_limit}_t}{2}, \qquad
p_t = \frac{\text{gas\_used}_t - \text{target\_gas}_t}{\text{target\_gas}_t}
$$

Interpretation:
- $p_t > 0$: congestion pressure is high.
- $p_t < 0$: pressure is lower.

### 2) Base fee momentum

$$
m_t = \log(\text{baseFee}_t) - \log(\text{baseFee}_{t-1})
$$

And optionally acceleration:

$$
a_t = m_t - m_{t-1}
$$

Interpretation:
- Positive momentum often indicates rising demand pressure.

### 3) Transaction surprise

$$
u_t = \text{zscore}\left(\text{txCount}_t - MA_k(\text{txCount})\right)
$$

Interpretation:
- Captures bursts beyond local baseline.

### 4) Latent backlog proxy (pseudo-mempool)

$$
b_t = \max\left(0, \rho b_{t-1} + \alpha p_t + \beta u_t\right)
$$

Interpretation:
- $b_t$ is a hidden-pressure state estimate, not true mempool size.
- Keep coefficients fixed during an experiment.

## Recommended state for offline RL

A compact state vector:

$$
s_t = [\text{baseFee}_t, p_t, m_t, a_t, u_t, b_t, d_t]
$$

where $d_t$ is deadline pressure (remaining time ratio).

## What is "research-grade" here

### Must state clearly in report
- We do not observe true mempool.
- We use pseudo-mempool estimators from block-level signals.
- This is a partial-observation setup.

### Must run ablations
- Without $b_t$.
- With $b_t$ only.
- With $b_t + m_t + a_t$.

If ablations do not improve stability or objective, keep simpler state.

## Reading order (papers and references)

### 1) Mechanism and fee dynamics (foundation)
- EIP-1559 specification (Ethereum docs / EIP repo).
- Tim Roughgarden et al., economic analyses of EIP-1559 and transaction fee mechanisms.

### 2) Gas/fee prediction from block-level data
- Search keywords:
  - "Ethereum gas fee prediction"
  - "EIP-1559 base fee forecasting"
  - "block-level features gas estimation"

Purpose:
- Learn which observable features best proxy demand pressure.

### 3) RL under partial observability
- Search keywords:
  - "POMDP reinforcement learning"
  - "DRQN partial observability"
  - "offline RL partial observability"

Purpose:
- Justify why hidden state estimation is acceptable without direct mempool.

### 4) Offline RL for conservative policy improvement
- Core algorithms to compare:
  - IQL
  - CQL
  - TD3+BC

Purpose:
- Understand why conservative methods are preferred on static datasets.

## Concrete interpretation for this project

Current status in this repo is good for engineering validation, but still pre-research because action is proxy-based.

Minimum upgrade path:
1. Keep pseudo-mempool features for state.
2. Replace proxy action with real logged action source.
3. Freeze reward equation and coefficients.
4. Run time-based split + OPE + confidence intervals.

That is the shortest path from "pipeline works" to "research claim is defensible".
