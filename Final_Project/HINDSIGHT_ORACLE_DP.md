# Theoretical Lower Bound via Hindsight Optimization (The God-View Oracle)

To establish an absolute performance ceiling for the reinforcement learning agent, we formulate a deterministic optimization baseline known as **Hindsight Optimization**. By granting the algorithm perfect forward-looking knowledge of the entire gas price trajectory $\mathbf{G} = (g_0, g_1, \dots, g_T)$ across the episode, the stochastic Markov Decision Process collapses into a **Deterministic Sequential Decision-Making** problem.

The singular exact method for extracting the global optimum within this non-convex landscape (due to the indicator function on base overheads) is **Backward Induction Dynamic Programming**.

## 1. Immediate Cost Function

Rather than maximizing a reward signal, Dynamic Programming natively minimizes the aggregate sequential cost. The immediate cost incurred at step $t$ for executing a batch volume of $n_t \in [0, Q_t]$ is formulated as:

$$c(Q_t, n_t, g_t) = \underbrace{g_t \cdot \left( C_{base} \cdot \mathbb{1}_{(n_t>0)} + C_{mar} \cdot n_t \right)}_{\text{Execution Constraint}} + \underbrace{\omega \cdot (Q_t - n_t + W_t)}_{\text{Delay/Holding Penalty}}$$

## 2. Deterministic Bellman Equation

Let $V_t^*(Q)$ represent the minimal cumulative cost from time $t$ to the terminal horizon $T$, assuming the system possesses a pending queue of size $Q$. The recursive Bellman equation is defined as:

$$V_t^*(Q) = \min_{n_t \in \{0, \dots, Q\}} \Big\{ c(Q, n_t, g_t) + V_{t+1}^*(Q - n_t + W_t) \Big\}$$

**Strict Boundary Condition (Terminal State $t=T$):**
At the final boundary, the system is strictly mandated to clear all pending transactions to prevent SLA violation (Fatal Penalty). Consequently, the isolated valid action is $n_T = Q$:
$$V_T^*(Q) = c(Q, Q, g_T)$$

## 3. Algorithm Implementation

The resolution algorithm consists of a dual-pass matrix operation across the temporal and queue state limits ($T \times Q_{max}$).

### Phase 1: Backward Induction
*   Initialize the boundary values $V_T^*(Q)$ across all possible $Q \in [0, Q_{max}]$.
*   Iterate backwards in time $t = T-1, \dots, 0$.
*   For each state permutation $(t, Q)$, evaluate the objective sequence across all possible execution volumes $n_t$.
*   Persist the exact optimal action into the objective policy map: $\pi_t^*(Q) = \arg\min (\dots)$

### Phase 2: Forward Tracing
*   Initialize temporal iteration at $t=0$ utilizing the factual starting queue $Q_0 = 0$.
*   Retrieve the optimal batch decision from the policy map: $n_0^* = \pi_0^*(Q_0)$.
*   Determine the sequential state projection: $Q_{t+1} = Q_t - n_t^* + W_t$.
*   The final output yields the theoretical optimal trajectory (The Oracle Path).

## 4. Vectorized Computational Implementation (Python Array Ops)

Due to the $O(T \cdot Q_{max}^2)$ polynomial bounds, processing large horizons on CPU boundaries dictates strict tensor vectorization. The exact implementation mechanism is supplied below:

```python
import numpy as np

def compute_god_view_trajectory(
    gas_prices: np.ndarray, 
    incoming_requests: np.ndarray, 
    Q_max: int, 
    C_base: float, 
    C_mar: float, 
    omega: float
):
    """
    Computes absolute minimal cost execution trajectory via Vectorized Backward Induction.
    """
    T = len(gas_prices)
    
    # State-Value Matrix V(t, q)
    V = np.full((T, Q_max + 1), np.inf, dtype=np.float32)
    # Action-Policy Matrix pi*(t, q)
    Policy = np.zeros((T, Q_max + 1), dtype=np.int32)
    
    # --- PHASE 1: BACKWARD INDUCTION ---
    # Terminal boundary enforcement 
    Q_array = np.arange(Q_max + 1)
    cost_T = gas_prices[T-1] * (C_base * (Q_array > 0) + C_mar * Q_array)
    V[T-1, :] = cost_T
    Policy[T-1, :] = Q_array # Only legally permitted deterministic action: execute all
    
    for t in range(T - 2, -1, -1):
        g_t = gas_prices[t]
        w_t = incoming_requests[t]
        
        for Q in range(Q_max + 1):
            n_array = np.arange(0, Q + 1)
            
            # Vectorized cost extraction
            exec_cost = g_t * (C_base * (n_array > 0) + C_mar * n_array)
            Q_next = np.clip(Q - n_array + w_t, 0, Q_max)
            delay_cost = omega * Q_next
            
            # Global objective cost 
            total_cost = exec_cost + delay_cost + V[t+1, Q_next]
            best_idx = np.argmin(total_cost)
            
            V[t, Q] = total_cost[best_idx]
            Policy[t, Q] = n_array[best_idx]
            
    # --- PHASE 2: FORWARD TRACING ---
    optimal_n = np.zeros(T, dtype=np.int32)
    optimal_a = np.zeros(T, dtype=np.float32) 
    
    current_Q = 0 
    for t in range(T):
        current_Q = min(current_Q, Q_max)
        
        n_star = Policy[t, current_Q]
        optimal_n[t] = n_star
        
        # Map back to generalized continuous continuous policy scalar
        optimal_a[t] = (n_star / current_Q) if current_Q > 0 else 0.0
        current_Q = current_Q - n_star + incoming_requests[t]
        
    return optimal_a, optimal_n, V[0, 0]
```

## 5. Deployment within Offline RL Context

The primary function of this formulation is not algorithm replacement, but **Offline Dataset Enhancement**. By running this operation dynamically across the historical gas history map, we extract optimal traces ($a_t^*$). 

These sequences are systematically interlinked back into the public blockchain logged dataset, forming a synthetic Multi-Policy distribution mechanism (e.g., 50% optimal oracle traces, 30% baseline human routing, 20% suboptimal edge cases). Through this procedure, the Implicit Q-Learning (IQL) execution paradigm naturally internalizes these macro-level smoothing operations, successfully mimicking the Oracle properties whilst remaining fundamentally bounded solely by localized Markov states and unprivileged future observation windows.
