import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from .config import TransitionBuildConfig
import json

class CharityGasEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, episode_df: pd.DataFrame, config: TransitionBuildConfig):
        super().__init__()
        self.episode_df = episode_df.reset_index(drop=True)
        self.config = config
        
        # Action: Continuous ratio [0, 1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space: 11 dimensions
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        
        self.current_step = 0
        self.max_step = len(episode_df)
        self.queue_size = 0.0
        self.time_to_deadline = 0.0
        
        # Normalization (optional)
        self.mins = None
        self.maxs = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        row = self.episode_df.iloc[0]
        state_val = row["state"]
        if isinstance(state_val, str):
            state_val = json.loads(state_val.replace("'", "\""))
        state_arr = np.array(state_val, dtype=np.float32)
        
        # Extract initial latent state
        self.queue_size = float(row.get("queue_size", 0.0))
        self.time_to_deadline = float(row.get("time_to_deadline", 0.0))
        
        return state_arr, {}

    def step(self, action):
        if self.current_step >= self.max_step:
            return np.zeros(11, dtype=np.float32), 0.0, True, False, {}

        a_t = np.clip(action[0], 0.0, 1.0)
        
        # Fix Zeno's Paradox for Continuous Neural Networks
        # NNs asymptotically approach 1.0 but rarely output exactly 1.000.
        # If the AI predicts > 95% execution, it means "Clear the Queue".
        if a_t >= 0.95:
            a_t = 1.0
            
        row = self.episode_df.iloc[self.current_step]
        
        exec_cap = self.config.execution_capacity
        
        # Fix Floating point stability: use np.round to avoid floor(0.9999) -> 0
        executed_volume = min(np.round(a_t * self.queue_size), exec_cap)
        
        # Calculate remaining queue (pre-arrival of NEXT block) for the urgency penalty
        remaining_q = max(0.0, self.queue_size - executed_volume)
        
        # Advance steps and compute physically consistent queue recurrence (sync with builder)
        if self.current_step < self.max_step - 1:
            next_row = self.episode_df.iloc[self.current_step + 1]
            arrival_next = float(next_row.get("transaction_count", 0.0))
            self.queue_size = remaining_q + arrival_next
            self.time_to_deadline = float(next_row.get("time_to_deadline", 0.0))
        else:
            self.queue_size = remaining_q
            self.time_to_deadline = 0.0
        
        # 5. Reward Calculation (Algebraic mirror of Builder)
        gas_scale = self.config.gas_to_gwei_scale
        gas_t = float(row["gas_t"]) / gas_scale
        gas_ref = float(row.get("gas_reference", row["gas_t"])) / gas_scale
        
        C_base = self.config.C_base
        C_mar = self.config.C_mar

        # Efficiency
        R_eff = executed_volume * C_mar * (gas_ref - gas_t)
        R_overhead = C_base * gas_t * float(executed_volume > 0)
        execution_reward = (R_eff - R_overhead) / self.config.reward_scale
        
        # Urgency (based on remaining queue AFTER action but BEFORE next arrivals)
        beta = self.config.urgency_beta
        alpha = self.config.urgency_alpha
        max_time_h = float(self.config.episode_hours)
        t_denom = max_time_h * 3600.0 if self.time_to_deadline > 24 else max_time_h
        t_curr = float(row.get("time_to_deadline", 1.0))
        time_ratio = np.clip(t_curr / max(1e-6, t_denom), 0.0, 1.0)
        
        urgency_penalty = (beta / self.config.reward_scale) * remaining_q * np.exp(alpha * (1.0 - time_ratio))
        
        total_reward = execution_reward - urgency_penalty
        
        self.current_step += 1
        
        # 6. Terminal status
        out_of_data = (self.current_step >= self.max_step)
        terminated = bool(out_of_data)
        truncated = bool(out_of_data)
        
        # SLA Violation (at episode end)
        if terminated and self.queue_size > 0:
            penalty = (self.config.deadline_penalty / self.config.reward_scale)
            total_reward -= penalty

        info = {
            "executed": executed_volume,
            "q_t": self.queue_size,
            "cost": gas_t * executed_volume,
            "deadline_miss": bool(terminated and self.queue_size > 0),
            "reward_components": {
                "efficiency": R_eff,
                "overhead": R_overhead,
                "urgency": urgency_penalty
            }
        }
        
        # Construct next state (compute final physical state)
        if self.current_step < self.max_step:
            next_row = self.episode_df.iloc[self.current_step]
            next_state_val = next_row["state"]
            if isinstance(next_state_val, str):
                next_state_val = json.loads(next_state_val.replace("'", "\""))
            next_obs = np.array(next_state_val, dtype=np.float32).copy()
        else:
            row = self.episode_df.iloc[self.current_step - 1]
            state_val = row["state"]
            if isinstance(state_val, str):
                state_val = json.loads(state_val.replace("'", "\""))
            next_obs = np.array(state_val, dtype=np.float32).copy()

        # Inject physically consistent queue and time (synced with builder)
        if self.mins is not None and self.maxs is not None:
            denom = np.where((self.maxs - self.mins) == 0, 1.0, (self.maxs - self.mins))
            next_obs[8] = (self.queue_size - self.mins[8]) / denom[8]
            next_obs[9] = (self.time_to_deadline - self.mins[9]) / denom[9]
        else:
            next_obs[8] = self.queue_size
            next_obs[9] = self.time_to_deadline

        return next_obs, total_reward, terminated, truncated, info