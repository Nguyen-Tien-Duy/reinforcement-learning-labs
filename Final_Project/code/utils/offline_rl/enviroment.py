import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from .config import TransitionBuildConfig
import json

class CharityGasEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, episode_df: pd.DataFrame, config: TransitionBuildConfig, mins=None, maxs=None):
        super().__init__()
        self.config = config
        
        # 1. Action & Observation spaces (DISCRETE V6)
        self.action_space = spaces.Discrete(config.n_action_bins)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        
        # 2. Single Source of Truth from Schema
        from .schema import STATE_COLS, Q_IDX, T_IDX
        self._STATE_COLS = STATE_COLS
        self._Q_IDX = Q_IDX
        self._T_IDX = T_IDX
        
        # 3. Buffer Data as NumPy Arrays (Massive Speedup vs Pandas iloc)
        self.max_step = len(episode_df)
        self.gas_t_raw = episode_df["gas_t"].to_numpy(dtype=np.float32)
        
        gas_ref_col = "gas_reference" if "gas_reference" in episode_df.columns else "gas_t"
        self.gas_ref_raw = episode_df[gas_ref_col].to_numpy(dtype=np.float32)
        
        # Optimized Column extraction with defaults
        def _get_arr(col, default_val=0.0):
            if col in episode_df.columns:
                return episode_df[col].to_numpy(dtype=np.float32)
            return np.full(self.max_step, default_val, dtype=np.float32)

        arr_scale = float(getattr(config, "arrival_scale", 0.5))
        # DISCRETE PHYSICS: Arrivals must be integers
        self.arrival_arr = np.round(_get_arr("transaction_count") * arr_scale).astype(np.int64)
        self.t_deadline_arr = _get_arr("time_to_deadline")
        
        # Pre-scale gas values for efficiency
        gas_scale = self.config.gas_to_gwei_scale
        self.gas_t_gwei = self.gas_t_raw / gas_scale
        self.gas_ref_gwei = self.gas_ref_raw / gas_scale
        
        # Buffered State Matrix for observations
        self.states_matrix = episode_df[self._STATE_COLS].to_numpy(dtype=np.float32)
        
        self.current_step = 0
        
        # 4. Raw initialization values for physics (must be discrete)
        self.raw_q_initial = int(np.round(float(episode_df["queue_size"].iloc[0]))) if "queue_size" in episode_df.columns else 0
        self.raw_t_initial = float(episode_df["time_to_deadline"].iloc[0]) if "time_to_deadline" in episode_df.columns else 0.0
        
        self.queue_size = 0
        self.time_to_deadline = 0.0
        
        # Normalization: Strict Policy Enforced
        self.normalize_state = config.normalize_state
        if self.normalize_state:
            if mins is None or maxs is None:
                raise ValueError(
                    "CharityGasEnv: normalize_state=True requires mins/maxs to be provided at init. "
                    "Fail-fast policy: AI cannot perceive the environment without valid scaling parameters."
                )
            self.mins = np.array(mins, dtype=np.float32)
            self.maxs = np.array(maxs, dtype=np.float32)
        else:
            self.mins = None
            self.maxs = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Initialize latent state from buffered matrix at index 0
        state_arr = self.states_matrix[0].copy()
        
        # Sync latent variables with RAW initial values for physics simulation
        self.queue_size = self.raw_q_initial
        self.time_to_deadline = self.raw_t_initial
        
        # Injection: Normalized Queue and Time for initial observation
        if self.normalize_state:
            denom = np.where((self.maxs - self.mins) == 0, 1.0, (self.maxs - self.mins))
            # [SOTA] Apply Log-Scaling to queue before normalization
            log_q = np.log1p(float(self.queue_size))
            state_arr[self._Q_IDX] = (log_q - self.mins[self._Q_IDX]) / denom[self._Q_IDX]
            state_arr[self._T_IDX] = (self.time_to_deadline - self.mins[self._T_IDX]) / denom[self._T_IDX]
        
        return state_arr, {}

    def step(self, action):
        if self.current_step >= self.max_step:
            return np.zeros(11, dtype=np.float32), 0.0, True, False, {}

        # 1. Map Discrete Action ID → Execution Ratio
        action_id = int(action)
        action_id = max(0, min(action_id, self.config.n_action_bins - 1))
        ratio = self.config.action_bins[action_id]
            
        exec_cap = self.config.execution_capacity
        
        # 2. Execution Volume Logic (Must be Integer)
        executed_volume = int(min(np.floor(ratio * self.queue_size), exec_cap))
        remaining_q = max(0, self.queue_size - executed_volume)
        
        # 3. Advance steps & Compute Physically Consistent Queue (Buffered)
        curr_idx = self.current_step
        next_idx = curr_idx + 1
        
        if next_idx < self.max_step:
            arrival_next = self.arrival_arr[next_idx]
            self.queue_size = remaining_q + arrival_next
            self.time_to_deadline = self.t_deadline_arr[next_idx]
        else:
            self.queue_size = remaining_q
            self.time_to_deadline = 0.0
        
        # 4. Reward Calculation (Algebraic mirror of Builder - using Buffered scaled values)
        gas_t = self.gas_t_gwei[curr_idx]
        gas_ref = self.gas_ref_gwei[curr_idx]
        
        reward_scale = self.config.reward_scale

        # Efficiency (Spec-aligned: R_eff = [n * (ref - t) - C_base * t * (n > 0)] / s_g)
        s_g = self.config.gas_scaling_factor
        C_base = getattr(self.config, "C_base", 21000.0)
        has_exec = 1.0 if executed_volume > 0 else 0.0
        
        efficiency_savings = executed_volume * (gas_ref - gas_t)
        overhead_cost = (C_base / 1e9) * gas_t * has_exec
        
        R_eff = (efficiency_savings - overhead_cost) / s_g
        
        # Urgency (Spec 2.2 with Clipping)
        beta = self.config.urgency_beta
        alpha = self.config.urgency_alpha
        max_time_h = float(self.config.episode_hours)
        
        t_curr = self.t_deadline_arr[curr_idx]
        time_ratio = np.clip(t_curr / max(1e-6, max_time_h), 0.0, 1.0)
        
        # Urgency (Linear scale synced with V19 Build/Oracle)
        urgency_penalty = beta * remaining_q * np.exp(alpha * (1.0 - time_ratio))
        
        total_reward = R_eff - urgency_penalty
        
        self.current_step += 1
        
        # 5. Terminal status
        out_of_data = (self.current_step >= self.max_step)
        terminated = bool(out_of_data)
        truncated = bool(out_of_data)
        
        # Triple-Sync V27: Linear penalty — proportional to remaining queue.
        R_cat = 0.0
        if terminated and self.queue_size > 0:
            R_cat = self.config.deadline_penalty * self.queue_size
            
        # Final Assembly with Strict Scaling (V22)
        reward_scale = self.config.reward_scale
        total_reward = (R_eff - urgency_penalty - R_cat) / reward_scale
        

        info = {
            "executed": executed_volume,
            "q_t": self.queue_size,
            "cost": gas_t * executed_volume,
            "deadline_miss": bool(terminated and self.queue_size > 0),
            "reward_components": {
                "efficiency": R_eff,
                "urgency": urgency_penalty
            }
        }
        
        # 6. Next Observation construction (Buffered)
        # 6. Next Observation construction
        if self.current_step < self.max_step:
            next_obs = self.states_matrix[self.current_step].copy()
        else:
            # Standard RL practice: return zeros for the observation at terminal state
            next_obs = np.zeros_like(self.states_matrix[0])

        # Inject physically consistent queue and time (synced with builder)
        if self.mins is not None and self.maxs is not None:
            denom = np.where((self.maxs - self.mins) == 0, 1.0, (self.maxs - self.mins))
            # [SOTA] Apply Log-Scaling to queue before normalization
            log_q = np.log1p(float(self.queue_size))
            next_obs[self._Q_IDX] = (log_q - self.mins[self._Q_IDX]) / denom[self._Q_IDX]
            next_obs[self._T_IDX] = (self.time_to_deadline - self.mins[self._T_IDX]) / denom[self._T_IDX]
        else:
            # Nếu không có bộ chuẩn hóa, trả về giá trị vật lý thô (Dành cho DT có scaler nội bộ)
            next_obs[self._Q_IDX] = float(self.queue_size)
            next_obs[self._T_IDX] = float(self.time_to_deadline)

        return next_obs, total_reward, terminated, truncated, info