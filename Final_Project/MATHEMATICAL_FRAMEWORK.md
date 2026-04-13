# Mathematical Formulation of the Continuous-Action MDP

A Markov Decision Process (MDP) is formally defined by the tuple $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$. The following structure specifies the MDP for deadline-constrained transaction batching.

## 1. State Space ($\mathcal{S}$)

The state $s_t \in \mathcal{S}$ at time $t$ adheres to the Markov property, satisfying $P(s_{t+1} | s_1, ..., s_t) = P(s_{t+1} | s_t)$. The state vector decomposes into exogenous and endogenous components.

**Exogenous State (Market Variables):**
$$M_t = \begin{bmatrix} g_t \\ \nabla g_t \\ \nabla^2 g_t \\ p_t \\ u_t \\ b_t \end{bmatrix} \in \mathbb{R}^6$$
Where:
*   $g_t$: Current base fee.
*   $\nabla g_t = g_t - g_{t-1}$: Fee momentum.
*   $\nabla^2 g_t$: Fee acceleration.
*   $p_t, u_t, b_t$: Mempool approximation states (congestion pressure, transaction surprise, latent backlog).

**Endogenous State (Agent-Controlled Variables):**
$$E_t = \begin{bmatrix} Q_t \\ \tau_t \end{bmatrix} \in \mathbb{R}^2$$
Where:
*   $Q_t \in [0, Q_{max}]$: Current queue length (pending transactions).
*   $\tau_t \in [0, \mathcal{T}]$: Remaining time until the strict deadline.

The combined state vector is determined by $s_t = M_t \oplus E_t \in \mathbb{R}^8$.

## 2. Action Space ($\mathcal{A}$)

To exploit economies of scale in smart contract execution, the action space transitions from a binary set to a continuous scale:

$$a_t \in [0, 1]$$

Here, $a_t$ represents the percentage of the pending queue $Q_t$ selected for execution at step $t$. The actual number of processed requests evaluates to a discrete step function:
$$n_t = \lfloor a_t \cdot Q_t \rfloor \in \mathbb{N}$$

The discretization from continuous $a_t$ to discrete $n_t$ introduces non-differentiability, establishing RL as a mathematical necessity over standard gradient-based optimization.

## 3. Transition Dynamics ($\mathcal{P}$)

The transition probability function $\mathcal{P}(s_{t+1} | s_t, a_t)$ splits into independent sub-processes:

**Exogenous Dynamics:**
Gas price fluctuations follow a stochastic process independent of the agent's action $a_t$ (assuming negligible market impact for the charity domain):
$$M_{t+1} \sim f(M_t) + \epsilon_t$$

**Endogenous Dynamics:**
Determined purely by deterministic update rules:
*   Queue evolution: $Q_{t+1} = \max(0, Q_t - n_t) + W_t$
    ($W_t \sim \text{Poisson}(\lambda)$ indicates new incoming requests).
*   Time decay: $\tau_{t+1} = \max(0, \tau_t - \Delta t)$.

## 4. Reward Structure ($\mathcal{R}$)

The multi-objective reward function $r_t = R(s_t, a_t)$ manages the core trade-off between fiscal conservation and deadline enforcement:

$$r_t = R_{eff}(t) - R_{overhead}(t) - R_{delay}(t) - R_{fatal}(t)$$

**A. Marginal Efficiency Bonus:**
$$R_{eff}(t) = n_t \cdot (g_{ref} - g_t)$$
Where $g_{ref}$ is a moving-average baseline target. The system incentivizes high-volume execution ($n_t$) exactly when current costs drop below historical benchmarks.

**B. Fixed Overhead Penalty:**
$$R_{overhead}(t) = C_{base} \cdot g_t \cdot \mathbb{1}_{(n_t > 0)}$$
$C_{base}$ represents the base transaction gas limit (e.g., 21,000 gas). $\mathbb{1}$ activates strictly when a broadcast occurs. This term natively punishes frequent, low-volume dispatches, enforcing operational batching.

**C. Holding/Delay Cost:**
$$R_{delay}(t) = \omega \cdot Q_{t+1}$$
A persistent penalty ensuring the network does not needlessly hoard transactions when gas conditions are moderate.

**D. Structural Deadline Penalty:**
$$R_{fatal}(t) = \lambda_{fatal} \cdot \exp\left(-\kappa \cdot \frac{\tau_{t+1}}{\mathcal{T}}\right) \cdot \mathbb{1}_{(Q_{t+1} > 0)}$$
A steeply convex exponential penalty enforcing extreme risk aversion. As $\tau \to 0$, the penalty scales asymptotically to enforce complete queue clearing prior to the deadline breach.

## 5. Discount Factor ($\gamma$)

For defined 24-hour episodic bounds, the discount factor is rigidly set:
$$\gamma \approx 0.99$$
This high discounting mitigates immediate-reward myopia, ensuring cost savings achieved late in the 23rd hour propagate structural credit back to decisions made in the 1st hour.
