# Final Project Objective: Offline RL for Charity Gas Fee Optimization

## 1. Executive Goal

Build an Offline Reinforcement Learning agent that decides when to submit queued charity transactions on blockchain.

Primary objectives:
- Meet delivery deadlines for each transaction batch.
- Minimize total gas spending.

This is a sequential decision problem, not only a forecasting problem. The agent must balance:
- Waiting for potentially cheaper gas.
- Executing now to avoid deadline misses.

## 2. Why RL Instead of Prediction-Only Modeling

A prediction model answers:
- "What will gas look like in the next block?"

An RL policy answers:
- "Given the current situation, should we wait or execute now?"

The RL policy is decision-centric and explicitly optimizes long-term utility under constraints.

## 3. MDP Formulation

### 3.1 State

At time step $t$, define the state as

$$
s_t = [g_t, g_{t-1}, g_{t-2}, q_t, \tau_t, z_t],
$$

where:
- $g_t$: current gas price (or base fee).
- $q_t$: pending queue size.
- $\tau_t$: remaining time to deadline.
- $z_t$: optional context features (for example tx count, block size, trend features).

### 3.2 Action

Initial discrete action space:

$$
\mathcal{A} = \{0, 1\}
$$

- $a_t = 0$: wait.
- $a_t = 1$: execute all queued transactions.

Optional extension:
- partial execution actions (for example 25 percent, 50 percent, 75 percent).

### 3.3 Transition

The market is mostly exogenous, while the queue is action-dependent.

$$
q_{t+1} = \max(0, q_t - u(a_t, q_t)), \qquad
\tau_{t+1} = \max(0, \tau_t - 1).
$$

Important interpretation:
- $s_{t+1}$ does not have to be fully caused by the action.
- It is the next world state the agent must face.

### 3.4 Reward

A practical reward design:

$$
r_t = -c_t(a_t, g_t, q_t)
- \lambda_d \cdot \mathbf{1}[\tau_{t+1}=0 \land q_{t+1}>0]
- \lambda_q \cdot q_{t+1},
$$

with:

$$
c_t =
\begin{cases}
0, & a_t = 0 \\
g_t \cdot u(a_t, q_t), & a_t > 0
\end{cases}
$$

Interpretation:
- Pay execution cost when sending transactions.
- Large deadline penalty if queue remains when time is over.
- Optional queue penalty to discourage excessive delay.

### 3.5 Terminal Condition

$$
d_t = \mathbf{1}[q_{t+1}=0 \lor \tau_{t+1}=0].
$$

Episode ends when all pending transactions are cleared or deadline window expires.

## 4. Offline RL Learning Objective

Given a static dataset

$$
\mathcal{D} = \{(s_t, a_t, r_t, s_{t+1}, d_t, e_t, tstamp_t)\}_{t=1}^{N},
$$

learn policy $\pi$ that maximizes discounted return:

$$
\pi^* = \arg\max_{\pi} \; \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{T-1} \gamma^t r_t\right].
$$

Value-learning target (generic form):

$$
y_t = r_t + \gamma (1 - d_t) \max_{a'} Q(s_{t+1}, a').
$$

## 5. Required Offline Dataset Contract

To train Offline RL correctly, the transition dataset must include:

| Column | Role | Required |
|---|---|---|
| state | $s_t$ observation | Yes |
| action | $a_t$ taken by behavior policy | Yes |
| reward | $r_t$ from reward function | Yes |
| next_state | $s_{t+1}$ | Yes |
| done | terminal flag $d_t$ | Yes |
| episode_id | trajectory grouping key | Yes |
| timestamp | temporal ordering key | Yes |

Key rule:
- Raw time-series features alone are not sufficient for Offline RL.
- You must have behavior actions and a defined reward function.

## 6. Practical Conversion Plan for Current Data

1. Build state vectors from gas history and queue/deadline features.
2. Attach action logs from historical operation policy (or a clearly defined behavior rule).
3. Compute reward with a documented formula and fixed coefficients.
4. Construct next_state by time-shifting within each episode.
5. Set done and episode_id consistently.
6. Validate schema and temporal consistency before training.

## 7. Training Plan

Recommended first algorithm: IQL with d3rlpy (offline-friendly and stable).

Pipeline:
1. Load parquet transition data.
2. Run strict schema validation.
3. Train on train split.
4. Evaluate on time-based holdout split.
5. Compare against rule-based baseline (always execute now, or threshold policy).

## 8. Evaluation Metrics

Core metrics:

$$
\text{AvgCost} = \frac{\sum \text{gas\_cost}}{\#\text{completed batches}},
$$

$$
\text{DeadlineMissRate} = \frac{\#\text{missed batches}}{\#\text{total batches}},
$$

$$
\text{Savings\%} = 100 \times \frac{\text{Cost}_{baseline} - \text{Cost}_{RL}}{\text{Cost}_{baseline}}.
$$

Evaluation requirements:
- Use timeline split to avoid leakage.
- Report on unseen periods only.

## 9. Deliverables

- A trained policy that outputs wait or execute decisions from state input.
- A reproducible pipeline: parquet -> transition dataset -> train -> evaluate.
- A comparison report: RL policy vs baseline cost and deadline performance.

## 10. Scope in This Repository

Current status:
- Large parquet data already exists.
- Transition builder and validator are in progress.
- After transition schema is complete, model training and evaluation can start.

This document defines the technical objective for the Final Project Offline RL workflow.
