# Reward Design Specification for Deadline-Constrained Batching

## Scope

This document defines a scientifically grounded reward design for the project and clarifies how the current implementation compares to that target.

No code change is required to use this document as reporting and experiment guidance.

## 1. Is the current reward "correct"?

Short answer: it is a valid engineering baseline, but not yet the strongest research-grade formulation.

### What is currently implemented

Current reward in the transition builder can be interpreted as:

$$
R_t^{impl} = \mathbf{1}[a_t=1](\overline{g}_t - g_t)
- \lambda_{exec}\mathbf{1}[a_t=1]
- \lambda_q q_t
- \lambda_d \mathbf{1}[\text{deadline miss at terminal}]
$$

where:
- $g_t$: current gas.
- $\overline{g}_t$: rolling moving-average gas reference.
- $q_t$: waiting queue size proxy.
- $\lambda_d$: deadline penalty.

### What is good about it
- Has efficiency term for sending at favorable gas.
- Has queue pressure term (if $\lambda_q>0$).
- Has catastrophic terminal penalty for missed deadline.
- Is stable and easy to train.

### What is still missing for high-level research claim
- No explicit urgency curve near deadline (risk-aversion growth near cutoff).
- Efficiency term is not explicitly scaled by executed batch size $n_t$.
- No explicit normalization scale for gas delta across regimes.

## 2. Theory-backed component reward (recommended)

Use a 3-component design:

$$
R_t = R_{efficiency,t} + R_{urgency,t} + R_{catastrophe,t}
$$

### 2.1 Efficiency component (marginal utility)

$$
R_{efficiency,t} = \mathbf{1}[a_t=1] \cdot n_t \cdot \frac{\overline{g}_t^{(k)} - g_t}{s_g}
$$

- $n_t$: number executed at step $t$ (or executed ratio times queue).
- $\overline{g}_t^{(k)}$: moving-average baseline over window $k$.
- $s_g$: scaling constant for numerical stability (for example 10 Gwei).

Rationale:
- Rewards sending below baseline.
- Scales with batch utility but stays numerically controlled.

### 2.2 Urgency component (cost of delay)

Base linear form:

$$
R_{urgency,t} = -\beta q_t
$$

Risk-averse near-deadline multiplier:

$$
R_{urgency,t} = -\beta q_t \cdot \exp\left(\alpha\left(1-\frac{\tau_t}{D}\right)\right)
$$

- $\tau_t$: remaining time.
- $D$: deadline window.
- $\alpha>0$: urgency curvature.

Rationale:
- Captures waiting harm continuously.
- Increases pressure as deadline approaches.

### 2.3 Catastrophe component (hard SLA violation)

$$
R_{catastrophe,t} = -\lambda_d \cdot \mathbf{1}[\tau_t=0 \land q_t>0]
$$

Rationale:
- Explicitly penalizes deadline breach when queue remains.

## 3. Recommended default parameters (starting point)

For initial experiments on limited hardware:
- $\lambda_d = 100$
- $\beta = 0.01$
- $s_g = 10$
- Baseline gas target example: 30 Gwei equivalent if you use fixed-reference experiments

Equivalent practical expression:

$$
R_t \approx \mathbf{1}[a_t=1] \cdot n_t \cdot \frac{30-g_t}{10}
-0.01\,q_t
-100\,\mathbf{1}[\tau_t=0 \land q_t>0]
$$

Important:
- If gas is stored in Wei, convert to Gwei before applying this formula.

## 4. Evaluation protocol to defend scientifically

To justify reward design in front of a committee, report ablations:

1. Efficiency-only.
2. Efficiency + Urgency.
3. Efficiency + Urgency + Catastrophe.
4. With and without urgency exponential term.

And report:
- Average cost per completed batch.
- Deadline miss rate.
- Total return.
- Sensitivity to $\beta$, $\lambda_d$, and urgency curvature $\alpha$.

## 5. Interpretation for this project status

Current project status is:
- Engineering-valid reward baseline: yes.
- Research-optimal reward evidence: not yet complete.

Minimal path to research-grade claim:
1. Add explicit $n_t$ scaling in efficiency term.
2. Use urgency multiplier near deadline.
3. Run ablation table and sensitivity analysis.
4. Keep reward coefficients fixed across all model comparisons.

## 6. Reproducibility note

Store reward settings per run in one config file (YAML or JSON), including:
- reward formula version
- $\beta$, $\lambda_d$, $\alpha$, $s_g$
- gas baseline type (fixed 30 Gwei or moving average)
- unit conversion rule (Wei to Gwei)

Without this, reward comparison across runs is not scientifically reliable.
