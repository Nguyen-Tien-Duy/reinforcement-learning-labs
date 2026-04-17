# REINFORCEMENT LEARNING DATA SPECIFICATION (V23)

## 1. Overview
The **V23 Specification** defines a robust process for transforming raw Ethereum blockchain data into a "Strict Discrete" transition dataset suitable for Offline Reinforcement Learning. The primary goal is to ensure high correlation between the agent's internal state perception and the physical reality of the blockchain network, particularly concerning gas prices and transaction queue dynamics.

### Key Enhancements in V23:
- **Strict Discrete Physics**: All transaction counts and queue sizes are rounded to integers to prevent fractional "ghost" transactions.
- **Hindsight Oracle Integration**: Uses Dynamic Programming (DP) to label optimal decision paths.
- **Unified Normalization (V22 legacy)**: Uses physical global baseline (Min/Max) for observation scaling instead of per-dataframe statistics.

---

## 2. High-Level Architecture: The Build Pipeline
The transformation follows a 4-stage sequential pipeline:

1.  **Stage 1: Raw Pre-processing**:
    - Load historical blocks from Parquet.
    - Group blocks into time-based episodes (default: 24-hour windows).
    - Sort values strictly by `timestamp`.

2.  **Stage 2: Initial Feature & Action Extraction**:
    - Extract economic features (Gas Prices, Congestion).
    - Derive internal features (Momentum, Acceleration).
    - Map behavioral actions (gas used) to discrete Bin IDs.

3.  **Stage 3: Hindsight Oracle Optimization**:
    - Perform Backward Induction on each episode to find the optimal execution path.
    - Mix behavioral actions with Oracle actions based on custom ratios.

4.  **Stage 4: Physical Sync & Transition Packaging**:
    - **Crucial**: Recalculate the entire episode's queue sequence because actions have changed.
    - Formulate $s_{t+1}$ and calculate Rewards $r_t$.
    - Standardize to `(s, a, r, ns, done)` format.

---

## 3. Mathematical State Space (11 Dimensions)
The observation vector $s_t \in \mathbb{R}^{11}$ is defined as follows:

### Phase A: Gas History (Dims 0, 1, 2)
Preserves temporal context of gas prices.
- $s_{0, 1, 2} = [\ln(g_t), \ln(g_{t-1}), \ln(g_{t-2})]$ where $g$ is the base fee per gas.

### Phase B: Network Congestion (Dim 3)
Calculates the relative deviation from the target gas limit (target = 50% of gas limit).
$$p_t = \frac{\text{gas\_used}_t - \text{target\_gas}_t}{\text{target\_gas}_t}$$

### Phase C: Price Momentum & Acceleration (Dims 4, 5)
Detects the speed and direction of price change.
- **Momentum**: $m_t = \ln(g_t) - \ln(g_{t-1})$
- **Acceleration**: $a_t = m_t - m_{t-1}$

### Phase D: Arrival Surprise (Dim 6)
Measures how unusual the incoming transaction volume is relative to a rolling window (default: 128 blocks).
$$u_t = \frac{w_t - \mu_{w, 128}}{\sigma_{w, 128}}$$
where $w_t$ is the count of incoming transactions.

### Phase E: Persistent Backlog (Dim 7)
An exponentially weighted moving average (EWMA) of congestion and surprise to capture long-term pressure.
$$b_t = \max(0, \rho \cdot b_{t-1} + \alpha_b \cdot p_t + \beta_b \cdot u_t)$$
*(Settings: $\rho=0.95, \alpha_b=0.3, \beta_b=0.2$)*

### Phase F: Queue & Time Dynamics (Dims 8, 9)
Physical environment constraints.
- **Queue ($Q_t$):** Current number of transactions awaiting execution.
- **Time Left ($T_t$):** Remaining hours in the 24h episode.

### Phase G: Gas Reference (Dim 10)
Rolling average of gas price ($g_{ref}$) used for calculating relative execution efficiency.

---

## 4. Action Space Definition (Discrete Bins)
The continuous execution ratio $r \in [0, 1]$ is quantized into 5 discrete bins.
- **Bin 0**: 0% (Do nothing)
- **Bin 1**: 25%
- **Bin 2**: 50%
- **Bin 3**: 75%
- **Bin 4**: 100% (Execution at maximum capacity)

**Execution logic**: The number of transactions executed $n_t$ is:
$$n_t = \min(\lfloor \text{BinRatio}_a \cdot Q_t \rfloor, \text{Capacity})$$
Where $\text{Capacity}$ is typically set to 500 units per step.

---

## 5. Reward Function Specification
The reward $R_t$ is a sum of three components, scaled for training stability.
$$R_t = \frac{R_{eff} - R_{urg} - R_{cat}}{\text{reward\_scale}}$$

### 5.1 Efficiency Reward ($R_{eff}$)
Threwards the agent for executing when current gas is lower than the rolling reference.
$$R_{eff} = n_t \cdot \frac{g_{ref} - g_t}{S_g}$$
*( $S_g$ is a scaling factor, usually 10.0)*

### 5.2 Urgency Penalty ($R_{urg}$)
Penalizes keeping transactions in the queue, especially as the deadline approaches.
$$R_{urg} = \beta \cdot Q_{rem} \cdot e^{\alpha \cdot (1 - \tau)}$$
Where:
- $Q_{rem} = Q_t - n_t$ (remaining queue).
- $\tau = \frac{T_t}{24}$ (time ratio remaining).
- $\beta, \alpha$ are penalty weights (Urgency Beta/Alpha).

### 5.3 Catastrophe Penalty ($R_{cat}$)
Strict penalty for a "Deadline Miss" (remaining queue > 0 at the end of the 24h episode).
$$R_{cat} = \lambda_d \cdot \mathbb{I}(\text{is\_last\_step} \text{ and } Q_{rem} > 0)$$
*( $\lambda_d$ is the Deadline Penalty factor)*

---

## 6. Discrete Queue Physics
To ensure physical consistency, the internal simulator enforces strict integer arrivals and executions.
1.  **Arrivals**: $w_t = \lfloor \text{RawArrivals}_t \cdot \text{ArrivalScale} \rfloor$
2.  **State Step**:
    $$Q_{t+1} = \max(0, Q_t - n_t + w_{t+1})$$

---

## 7. Hindsight Oracle (The "God-View" Solver)
The Oracle solves for the sequence of actions $a_{0..T}$ that minimizes the total cost across an entire episode.

### Cost Function (Minimized):
$$J = \sum_{t=0}^{T} \left( \text{ExecutionCost}_t + \text{UrgencyPenalty}_t \right) + \text{CatastrophePenalty}$$

### Implementation:
- **Algorithm**: Backward Induction (Dynamic Programming).
- **State Space**: $V(t, q)$ stores the minimum cost-to-go from step $t$ with queue $q$.
- **Optimization**: Numba-compiled (`@njit`) for performance.

### Mix-Policy Logic:
- **Oracle Action**: Optimal $a^*$ from DP.
- **Suboptimal Action**: Inverted $a^*$ (e.g., if $a^*=4$, suboptimal is $0$).
- **Mix Ratio**: Default 50% Oracle, 30% Behavioral, 20% Suboptimal.

---

## 8. Data Dictionary (transitions_v23.parquet)
| Column | Type | Description |
| :--- | :--- | :--- |
| `action` | int32 | Discrete bin ID (0-4). |
| `reward` | float32 | Total scaled reward (Log-Symmetric). |
| `done` | int8 | 1 for last step, 0 otherwise. |
| `episode_id` | int64 | Unique ID for the 24h window. |
| `queue_size` | float32 | Physical queue size at start of step. |
| `time_to_deadline` | float32 | Hours remaining. |
| `s_gas_t0` | float32 | Normalized log gas price (current). |
| `s_queue` | float32 | Normalized queue size. |
| `policy_type` | int8 | 0: Behavioral, 1: Oracle, 2: Suboptimal. |
| `ns_*` | float32 | Next State values for all `s_*` features. |
