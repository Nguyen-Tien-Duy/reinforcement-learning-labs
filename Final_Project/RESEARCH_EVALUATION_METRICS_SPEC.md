# Research-Level Evaluation Metrics Specification

## Purpose

This document defines a research-grade evaluation protocol for the Offline RL project on deadline-constrained batching.

The goal is to avoid weak conclusions from training curves alone and to evaluate decision quality, constraint satisfaction, and risk with statistically defensible evidence.

## 1) Problem framing

The task is not pure reward maximization. It is a constrained optimization problem:

$$
\min \; \mathbb{E}[\text{Cost}] \quad \text{s.t.} \quad \text{DeadlineMissRate} \le \delta
$$

So evaluation must report both objective and constraint metrics.

## 2) Metric taxonomy

Use three metric groups in every experiment.

### Group A: Support and reliability metrics

#### A1. Coverage

$$
\text{Coverage} = \frac{1}{N}\sum_{t=1}^{N}\mathbf{1}[\hat a_t = a_t]
$$

Interpretation:
- Measures how often the evaluated policy agrees with logged behavior.
- Low coverage indicates strong extrapolation risk.

#### A2. Effective sample size for IS-style estimators

For normalized importance weights $\tilde w_i$:

$$
\text{ESS} = \frac{1}{\sum_i \tilde w_i^2}
$$

Interpretation:
- Low ESS means unstable OPE and high estimator variance.

### Group B: Primary task metrics

#### B1. ExpectedReturn (matched)

$$
\hat J_{\text{matched}} =
\frac{\sum_{t=1}^{N}\mathbf{1}[\hat a_t=a_t]r_t}
{\sum_{t=1}^{N}\mathbf{1}[\hat a_t=a_t]}
$$

Interpretation:
- Fast proxy metric.
- Must always be read together with Coverage.

#### B2. DeadlineMissRate (constraint)

Define episode-level miss flag $m_e \in \{0,1\}$.

$$
\text{DeadlineMissRate} = \frac{1}{|\mathcal{E}|}\sum_{e\in\mathcal{E}} m_e
$$

Interpretation:
- Primary reliability/safety metric.
- A policy with low cost but high miss rate is unacceptable.

#### B3. TotalCostPerEpisode (objective)

If executed volume $n_t$ is available:

$$
\text{Cost}_e = \sum_{t\in e}\mathbf{1}[a_t=1]\cdot g_t\cdot n_t
$$

If $n_t$ is unavailable, use explicit proxy and state it clearly:

$$
\text{Cost}^{proxy}_e = \sum_{t\in e}\mathbf{1}[a_t=1]\cdot g_t
$$

Report:

$$
\text{TotalCostPerEpisode} = \frac{1}{|\mathcal{E}|}\sum_{e\in\mathcal{E}}\text{Cost}_e
$$

#### B4. ActionRate

For binary action (wait or execute):

$$
\text{ActionRate} = \frac{1}{N}\sum_{t=1}^{N}\mathbf{1}[\hat a_t=1]
$$

Interpretation:
- Distinguishes aggressive vs conservative policies.
- Useful for debugging policy collapse.

### Group C: Risk and confidence metrics

#### C1. Return variance

$$
\text{VarReturn} = \text{Var}(R_e), \quad R_e = \sum_{t\in e} r_t
$$

#### C2. CVaR at level $\alpha$

$$
\text{CVaR}_{\alpha}(R) = \mathbb{E}[R \mid R \le q_{\alpha}(R)]
$$

Interpretation:
- Captures tail risk (worst-case episodes).
- Critical for deadline-sensitive systems.

#### C3. Confidence intervals

Use episode-level bootstrap for every key metric (at least 1000 resamples), report 95% CI.

## 3) OPE estimators required for research-grade claims

Training loss and matched metrics are not sufficient. Include at least two OPE estimators.

### E1. Weighted Importance Sampling (WIS)

For trajectory-level returns $G_i$ and importance ratios $w_i$:

$$
\hat J_{\text{WIS}} = \sum_i \tilde w_i G_i,
\quad \tilde w_i = \frac{w_i}{\sum_j w_j}
$$

### E2. Doubly Robust (DR)

$$
\hat J_{\text{DR}} = \frac{1}{n}\sum_{i=1}^{n}
\left[\hat V(s_i) + \rho_i\left(r_i + \gamma \hat V(s'_i) - \hat Q(s_i,a_i)\right)\right]
$$

### E3. Fitted Q Evaluation (FQE)

Fit an evaluation critic for target policy and compute policy value on initial-state distribution.

Interpretation:
- More robust than pure IS in high-dimensional settings.

## 4) Recommended final metric set for this project

Minimum publishable set:
1. Coverage
2. ExpectedReturn (matched)
3. DeadlineMissRate
4. TotalCostPerEpisode
5. ActionRate
6. WIS estimate with 95% CI
7. DR or FQE estimate with 95% CI
8. CVaR (risk)

Optional operational score (clearly marked as heuristic):

$$
\text{AdjustedReturn} = \text{Coverage} \times \hat J_{\text{matched}}
$$

Do not use AdjustedReturn as the only decision criterion.

## 5) Required baseline comparisons

Always compare against:
1. Execute-now policy.
2. Threshold policy.
3. Behavior cloning baseline.
4. Offline RL policy (IQL/CQL/TD3+BC variant).

A claim is strong only if RL dominates baselines on cost while keeping deadline miss rate controlled.

## 6) Evaluation protocol (step-by-step)

1. Build strict time-based split (train, validation, test).
2. Freeze reward formula and all coefficients before comparison.
3. Train candidate policies on train split only.
4. Select checkpoint on validation split.
5. Report final metrics on unseen test split.
6. Compute episode-level bootstrap confidence intervals.
7. Include ablations for reward components and key features.

## 7) Reporting template

Use a table with mean plus 95% CI:

- Coverage
- ExpectedReturn (matched)
- DeadlineMissRate
- TotalCostPerEpisode
- ActionRate
- WIS
- DR/FQE
- CVaR

Include a Pareto chart:
- x-axis: TotalCostPerEpisode
- y-axis: DeadlineMissRate
- one point per policy

This plot directly shows objective-constraint trade-off.

## 8) Interpretation rules

1. Never conclude from training loss alone.
2. Never report matched return without coverage.
3. Prefer policies that reduce cost and keep miss rate within constraint.
4. If OPE estimators disagree strongly, mark result as inconclusive.
5. If CI overlaps heavily with baseline, do not claim improvement.

## 9) Practical notes for 16GB RAM

1. Run coarse sweep first with matched metrics and coverage.
2. Keep only top checkpoints for full OPE (WIS plus DR/FQE).
3. Use episode-level batching and cached features for evaluation.
4. Save one run config file (YAML or JSON) for each experiment.

## 10) Evidence threshold for "research-level" statement

You may claim research-grade evaluation only when all are true:
- Time-based holdout protocol is fixed and documented.
- At least two OPE estimators are reported with 95% CI.
- Objective and constraint metrics are both improved or properly traded off.
- Risk metrics (variance or CVaR) are reported.
- Full reproducibility artifacts are available.
