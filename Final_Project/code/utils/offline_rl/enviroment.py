import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CharityGasEnv(gym.Env):
    def __init__(self, episode_df, config):
        super(CharityGasEnv, self).__init__()
        self.config = config
        
        # Action space: Continuous percentage [0.0, 1.0]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # State has 10 dimensions: gas_t, gas_t-1, gas_t-2, p_t, m_t, a_t, u_t, b_t, queue_size, time_to_deadline
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        # Episolde_df contain relay data of 1 day (24h)
        self.episode_df = episode_df.reset_index(drop=True)
        self.current_step = 0
        self.max_step = len(self.episode_df)
        
        # Dynamic types
        self.queue_size = 0.0
        # Calculate real time decrement per step
        # Example: 24 hours / 7200 steps = 0.00333 hours per step
        self.time_step_size = float(config.episode_hours) / max(1, self.max_step)


    def reset(self, seed = None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.queue_size = 0.0
        
        self.time_to_deadline = float(self.config.episode_hours)

        return self._get_obs(), {}

    def step(self, action):
        # Get current row
        row = self.episode_df.iloc[self.current_step]
        
        # Parse continuous action [0.0, 1.0] from array
        if isinstance(action, (np.ndarray, list)):
            a_t = np.clip(float(action[0]), 0.0, 1.0)
        else:
            a_t = np.clip(float(action), 0.0, 1.0)
        
        # New transaction arrive
        arrival_t = 1.0 # We assume that 1 transaction arrive every step
        
        # Execute continuous volume n_t = floor(a_t * Q_t)
        executed_volume = float(np.floor(a_t * self.queue_size))
        self.queue_size = self.queue_size - executed_volume + arrival_t

        # Equation 3: tau_{t+1} = max(0, tau_t - time_step_size)
        self.time_to_deadline = max(0.0, self.time_to_deadline - self.time_step_size)

        # Time jump
        self.current_step += 1

        # Equation 4: Terminal flags (done_t)
        # Unlike discrete, episode only ends when we reach the deadline naturally.
        terminated = bool(self.time_to_deadline <= 0)
        out_of_data = (self.current_step >= self.max_step)
        truncated = bool(out_of_data)

        # Equation 5: Deadline miss (m_e)
        deadline_miss = bool(self.queue_size > 0 and self.time_to_deadline <= 0)

        # Component 1: Efficiency (marginal utility) with unit conversion (Gwei)
        gas_scale = getattr(self.config, "gas_to_gwei_scale", 1e9)
        gas_t = float(row["gas_t"]) / gas_scale
        gas_ref = float(row.get("gas_reference", row["gas_t"])) / gas_scale
        
        C_base = getattr(self.config, "C_base", 21000.0)
        C_mar = getattr(self.config, "C_mar", 15000.0)

        R_eff = executed_volume * C_mar * (gas_ref - gas_t)
        R_overhead = C_base * gas_t * float(executed_volume > 0)
        execution_reward = R_eff - R_overhead
        
        # Component 2: Urgency (cost of delay, risk-averse near deadline)
        beta = getattr(self.config, "urgency_beta", 0.01)
        alpha = getattr(self.config, "urgency_alpha", 3.0)
        max_time = float(self.config.episode_hours)
        time_ratio = min(1.0, max(0.0, self.time_to_deadline / max_time))
        
        urgency_penalty = beta * self.queue_size * np.exp(alpha * (1.0 - time_ratio))

        reward = execution_reward - urgency_penalty

        # Component 3: Catastrophe (hard SLA violation)
        if deadline_miss:
            reward -= self.config.deadline_penalty

        # Scale Reward exacly the same with building dataset
        reward_scale = getattr(self.config, "reward_scale", 1e6)
        reward = reward / reward_scale

        info = {
            "deadline_miss": deadline_miss,
            "cost": float(gas_t * (C_base * float(executed_volume > 0) + C_mar * executed_volume))
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        # Get the current row
        idx = min(self.current_step, self.max_step - 1)
        row = self.episode_df.iloc[idx]
        
        # GET PRE-NORMALIZED STATE ARRAY FROM HISTORICAL LOGS (Crucial!)
        # In the parquet file, the "state" column was already normalized (0-1) by build_state_action.py
        base_state = np.array(row["state"], dtype=np.float32).copy()
        
        # The only issue: Simulator's Queue may be higher or lower than historical logs.
        # We need to calculate the Compression Ratio (Max Queue) to scale self.queue_size accordingly.
        max_raw_q = self.episode_df["queue_size"].max()
        
        if max_raw_q > 0:
            # Find the row with the largest queue to see how much the AI compressed it
            max_row = self.episode_df[self.episode_df["queue_size"] == max_raw_q].iloc[0]
            # State: [gas0, gas1, gas2, p_t, m_t, a_t, u_t, b_t, queue, time]. Queue is at index 8
            norm_val = max_row["state"][8] 
            queue_max_estimate = max_raw_q / (norm_val + 1e-8)
            
            # Scale down the Simulator's virtual Queue using that ratio
            norm_queue = self.queue_size / queue_max_estimate
        else:
            norm_queue = 0.0
            
        # Overwrite the current Queue into the State array
        base_state[8] = norm_queue

        # Overwrite the current Time-to-Deadline into the State array
        # Normalize time the same way: divide by episode_hours (24) to get range [0, 1]
        base_state[9] = self.time_to_deadline / float(self.config.episode_hours)

        return base_state

        
         
        

        
        