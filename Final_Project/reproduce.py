from __future__ import annotations

from dataclasses import dataclass
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import d3rlpy
import gymnasium as gym
from gymnasium import spaces

# ─── SCHEMA CONSTANTS (Locked Source of Truth) ──────────────────────────────
STATE_COLS = [
    "s_gas_t0", "s_gas_t1", "s_gas_t2", "s_congestion", "s_momentum", "s_accel",
    "s_surprise", "s_backlog", "s_queue", "s_time_left", "s_gas_ref"
]
Q_IDX = 8
T_IDX = 9

@dataclass(frozen=True)
class TransitionBuildConfig:
    # --- CANONICAL PHYSICS (Synced with config.py for evaluation) ---
    arrival_scale: float = 0.1          # Evaluation config (config.py)
    deadline_penalty: float = 1000000.0 # Evaluation config (config.py)
    episode_hours: int = 24             
    history_window: int = 3             
    
    # --- ENVIRONMENT CONSTANTS & ECONOMICS ---
    C_base: float = 21000.0             
    C_mar: float = 15000.0              
    gas_scaling_factor: float = 10.0    
    gas_to_gwei_scale: float = 1e9      
    execution_capacity: float = 500.0   
    
    # --- OPTIONAL METADATA & COLUMNS ---
    timestamp_col: str = "timestamp"
    gas_col: str = "base_fee_per_gas"
    action_col: str = "action"
    queue_col: str = "queue_size"
    gas_used_col: str = "gas_used"
    gas_limit_col: str = "gas_limit"
    transaction_count_col: str = "transaction_count"
    
    # --- ECONOMICS & URGENCY ---
    gas_reference_window: int = 128
    normalize_state: bool = False       # Evaluation: d3rlpy has internal scaler
    urgency_alpha: float = 1.0          # Evaluation config (config.py)
    urgency_beta: float = 0.0001        # Evaluation config (config.py)
    reward_scale: float = 1.0           

    # --- DISCRETE ACTION SPACE (V6) ---
    n_action_bins: int = 5
    action_bins: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)

class CharityGasEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, episode_df: pd.DataFrame, config: TransitionBuildConfig, mins=None, maxs=None):
        super().__init__()
        self.config = config
        self.action_space = spaces.Discrete(config.n_action_bins)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        
        self.max_step = len(episode_df)
        self.gas_t_raw = episode_df["gas_t"].to_numpy(dtype=np.float32)
        gas_ref_col = "gas_reference" if "gas_reference" in episode_df.columns else "gas_t"
        self.gas_ref_raw = episode_df[gas_ref_col].to_numpy(dtype=np.float32)
        
        def _get_arr(col, default_val=0.0):
            if col in episode_df.columns:
                return episode_df[col].to_numpy(dtype=np.float32)
            return np.full(self.max_step, default_val, dtype=np.float32)

        arr_scale = float(getattr(config, "arrival_scale", 0.1))
        self.arrival_arr = np.round(_get_arr("transaction_count") * arr_scale).astype(np.int64)
        self.t_deadline_arr = _get_arr("time_to_deadline")
        
        gas_scale = self.config.gas_to_gwei_scale
        self.gas_t_gwei = self.gas_t_raw / gas_scale
        self.gas_ref_gwei = self.gas_ref_raw / gas_scale
        self.states_matrix = episode_df[STATE_COLS].to_numpy(dtype=np.float32)
        
        self.current_step = 0
        self.raw_q_initial = int(np.round(float(episode_df["queue_size"].iloc[0]))) if "queue_size" in episode_df.columns else 0
        self.raw_t_initial = float(episode_df["time_to_deadline"].iloc[0]) if "time_to_deadline" in episode_df.columns else 0.0
        self.queue_size = 0
        self.time_to_deadline = 0.0
        
        self.normalize_state = config.normalize_state
        if self.normalize_state:
            if mins is None or maxs is None:
                raise ValueError("CharityGasEnv: normalize_state=True requires mins/maxs.")
            self.mins = np.array(mins, dtype=np.float32)
            self.maxs = np.array(maxs, dtype=np.float32)
        else:
            self.mins = None
            self.maxs = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        state_arr = self.states_matrix[0].copy()
        self.queue_size = self.raw_q_initial
        self.time_to_deadline = self.raw_t_initial
        
        if self.normalize_state:
            denom = np.where((self.maxs - self.mins) == 0, 1.0, (self.maxs - self.mins))
            log_q = np.log1p(float(self.queue_size))
            state_arr[Q_IDX] = (log_q - self.mins[Q_IDX]) / denom[Q_IDX]
            state_arr[T_IDX] = (self.time_to_deadline - self.mins[T_IDX]) / denom[T_IDX]
        return state_arr, {}

    def step(self, action):
        if self.current_step >= self.max_step:
            return np.zeros(11, dtype=np.float32), 0.0, True, False, {}

        action_id = max(0, min(int(action), self.config.n_action_bins - 1))
        ratio = self.config.action_bins[action_id]
        executed_volume = int(min(np.floor(ratio * self.queue_size), self.config.execution_capacity))
        remaining_q = max(0, self.queue_size - executed_volume)
        
        curr_idx = self.current_step
        next_idx = curr_idx + 1
        if next_idx < self.max_step:
            self.queue_size = remaining_q + self.arrival_arr[next_idx]
            self.time_to_deadline = self.t_deadline_arr[next_idx]
        else:
            self.queue_size = remaining_q
            self.time_to_deadline = 0.0
        
        gas_t = self.gas_t_gwei[curr_idx]
        gas_ref = self.gas_ref_gwei[curr_idx]
        s_g = self.config.gas_scaling_factor
        C_base = getattr(self.config, "C_base", 21000.0)
        has_exec = 1.0 if executed_volume > 0 else 0.0
        
        efficiency_savings = executed_volume * (gas_ref - gas_t)
        overhead_cost = (C_base / 1e9) * gas_t * has_exec
        R_eff = (efficiency_savings - overhead_cost) / s_g
        
        beta = self.config.urgency_beta
        alpha = self.config.urgency_alpha
        max_time_h = float(self.config.episode_hours)
        t_curr = self.t_deadline_arr[curr_idx]
        time_ratio = np.clip(t_curr / max(1e-6, max_time_h), 0.0, 1.0)
        urgency_penalty = beta * remaining_q * np.exp(alpha * (1.0 - time_ratio))
        
        self.current_step += 1
        terminated = (self.current_step >= self.max_step)
        R_cat = self.config.deadline_penalty * self.queue_size if (terminated and self.queue_size > 0) else 0.0
        
        total_reward = (R_eff - urgency_penalty - R_cat) / self.config.reward_scale
        
        info = {"executed": executed_volume, "q_t": self.queue_size, "cost": gas_t * executed_volume, "deadline_miss": bool(terminated and self.queue_size > 0)}
        
        if self.current_step < self.max_step:
            next_obs = self.states_matrix[self.current_step].copy()
        else:
            next_obs = np.zeros(11, dtype=np.float32)

        if self.mins is not None and self.maxs is not None:
            denom = np.where((self.maxs - self.mins) == 0, 1.0, (self.maxs - self.mins))
            log_q = np.log1p(float(self.queue_size))
            next_obs[Q_IDX] = (log_q - self.mins[Q_IDX]) / denom[Q_IDX]
            next_obs[T_IDX] = (self.time_to_deadline - self.mins[T_IDX]) / denom[T_IDX]
        else:
            # Raw injection: dataset s_* columns are pre-normalized
            next_obs[Q_IDX] = float(self.queue_size)
            next_obs[T_IDX] = float(self.time_to_deadline)
            
        return next_obs, total_reward, terminated, False, info

# ======================================================================
# Xác định thư mục gốc của dự án (Project Root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 1. Dữ liệu đánh giá (Tập Test 20% cuối cùng - 242 episodes)
DATA_PATH = os.path.join(PROJECT_ROOT, 'Final_Project', 'Data', 'transitions_v33_L2_Batching_RAW.parquet')

# 2. Model Chiến Thắng (SOTA - Huấn luyện với Reward Shaping đúng đắn)
MODEL_PATH = os.path.join(PROJECT_ROOT, 'd3rlpy_logs', 'DiscreteCQL_V6_20260428_0426_20260428042630', 'model_40000.d3')

# 3. Parameters chuẩn hóa trạng thái (Normalization params)
NORM_PATH = os.path.join(PROJECT_ROOT, 'Final_Project', 'Data', 'state_norm_params.json')

# Cấu hình Safety Layer (Ép bán tháo khi còn <= 20% thời gian)
SAFETY_THRESHOLD = 0.20
# ======================================================================

def main():
    print(f"[{'*'*60}]")
    print(" SCRIPT TÁI LẬP KẾT QUẢ SOTA (STATE-OF-THE-ART)")
    print("Chiến lược: CQL model_160000 + Safety Layer 20%")
    print(f"[{'*'*60}]\n")

    # 1. Kiểm tra sự tồn tại của file
    for path, name in [(DATA_PATH, "Data"), (MODEL_PATH, "Model"), (NORM_PATH, "Norm Params")]:
        if not os.path.exists(path):
            print(f" LỖI: Không tìm thấy file {name} tại {path}")
            return
    print(" Đã tìm thấy tất cả các file đóng băng (Data, Model, Norm Config).")

    # 2. Load dữ liệu
    print("⏳ Đang load dữ liệu...")
    df = pd.read_parquet(DATA_PATH)
    unique_eps = sorted(df['episode_id'].unique())
    # Lấy 20% episodes cuối cùng làm tập test (hold-out)
    test_ids = unique_eps[int(len(unique_eps) * 0.8):]
    ep_list = [d.reset_index(drop=True) for _, d in df[df['episode_id'].isin(test_ids)].groupby('episode_id')]
    
    print(f" Đã load {len(ep_list)} episodes Test (Từ ID {test_ids[0]} đến {test_ids[-1]}).")

    # 3. Load cấu hình và Model
    print("⏳ Đang khởi tạo môi trường và Model...")
    config = TransitionBuildConfig()
    # Note: d3rlpy model đã có observation_scaler nội bộ (MinMax)
    # Không cần load norm params - truyền None để env inject raw values
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = d3rlpy.load_learnable(MODEL_PATH, device=device)
    print(f"Model load thành công trên thiết bị: {device}")

    # 4. Chạy mô phỏng đánh giá
    print("\n Đang mô phỏng các chiến lược trên toàn bộ tập Test. Vui lòng đợi...\n")
    
    greedy_costs = []
    cql_costs = []
    cql_misses = []
    
    for i, ep_df in enumerate(ep_list):
        if (i + 1) % 50 == 0:
            print(f"   Đã xử lý {i + 1}/{len(ep_list)} episodes...")

        # --- A. Đánh giá Greedy (Baseline) ---
        env = CharityGasEnv(ep_df, config, mins=None, maxs=None)
        env.reset()
        done = False
        gc = 0.0
        while not done:
            # Action 4: Xả 100% hàng ngay lập tức
            _, _, ter, tru, info = env.step(4)
            gc += info.get('cost', 0.0)
            if ter or tru:
                done = True
        greedy_costs.append(gc)
        
        # --- B. Đánh giá CQL + Safety Layer (SOTA) ---
        env = CharityGasEnv(ep_df, config, mins=None, maxs=None)
        obs, _ = env.reset()
        done = False
        cc = 0.0
        miss = False
        
        while not done:
            time_ratio = env.time_to_deadline / env.config.episode_hours
            
            # Kích hoạt Safety Layer
            if time_ratio < SAFETY_THRESHOLD:
                action = 4 # Ép xả 100%
            else:
                # CQL Model dự đoán
                obs_c = np.ascontiguousarray(obs.reshape(1, -1), dtype=np.float32)
                action = int(model.predict(obs_c)[0])
                
            obs, _, ter, tru, info = env.step(action)
            cc += info.get('cost', 0.0)
            
            if ter or tru:
                miss = info.get('deadline_miss', False)
                done = True
                
        cql_costs.append(cc)
        cql_misses.append(1 if miss else 0)

    # 5. Tổng hợp kết quả
    mean_greedy = np.mean(greedy_costs)
    mean_cql = np.mean(cql_costs)
    miss_rate = np.mean(cql_misses) * 100
    savings_pct = ((mean_greedy - mean_cql) / mean_greedy) * 100

    print(f"\n[{'='*60}]")
    print(f"KẾT QUẢ CUỐI CÙNG TRÊN TẬP TEST ĐỘC LẬP ({len(ep_list)} episodes)")
    print(f"[{'='*60}]")
    print(f"1. Greedy Baseline:    {mean_greedy:>10,.0f} Gwei")
    print(f"2. CQL + Safety 20%:   {mean_cql:>10,.0f} Gwei")
    print(f"3. Deadline Miss Rate: {miss_rate:>10.1f} %")
    print("-" * 62)
    
    if mean_cql < mean_greedy:
        print(f" HIỆU NĂNG: CQL tiết kiệm được +{savings_pct:.1f}% chi phí Gas so với Greedy!")
    else:
        print(f" HIỆU NĂNG: CQL đắt hơn Greedy {-savings_pct:.1f}%.")
    
    print(f"[{'='*60}]\n")

if __name__ == "__main__":
    main()
