import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Cấu hình Rời rạc hóa (Discretization)
BINS = [4, 4, 10, 10] 
MIN_VALS = np.array([-4.8, -4.0, -0.418, -4.0]) # Thu hẹp giới hạn vận tốc để tập trung vùng trung tâm
MAX_VALS = np.array([4.8, 4.0, 0.418, 4.0])
NUM_STATES = np.prod(BINS)

def discretize(state):
    ratio = (state - MIN_VALS) / (MAX_VALS - MIN_VALS)
    indices = (ratio * BINS).astype(int)
    indices = np.clip(indices, 0, np.array(BINS) - 1)
    return tuple(indices)

def state_to_int(state_tuple):
    idx = 0
    multiplier = 1
    for dim_size, val in zip(reversed(BINS), reversed(state_tuple)):
        idx += val * multiplier
        multiplier *= dim_size
    return idx

# 2. Chính sách cố định (Heuristic)
def simple_policy(state):
    pole_angle = state[2]
    return 0 if pole_angle < 0 else 1

# 3. Ground Truth Generation
def get_ground_truth(env, episodes=60000, alpha=0.01, gamma=0.99):
    v_true = np.zeros(NUM_STATES)
    print("--- Đang tạo ĐÁP ÁN CHUẨN (MC - 60,000 episodes) ---")
    
    for ep in range(episodes):
        state, _ = env.reset()
        trajectory = []
        done = False
        while not done:
            s_idx = state_to_int(discretize(state))
            action = simple_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            trajectory.append((s_idx, reward))
            state = next_state
        
        G = 0
        visited = set()
        for s_t, r_t in reversed(trajectory):
            G = r_t + gamma * G
            if s_t not in visited:
                v_true[s_t] += alpha * (G - v_true[s_t])
                visited.add(s_t)
        if (ep + 1) % 20000 == 0: print(f"  > Done {ep+1}/{episodes}...")
    return v_true

# 4. Deep-dive Comparison
def run_stability_analysis(env, episodes, initial_alpha, gamma, v_true):
    v_td = np.zeros(NUM_STATES)
    v_mc = np.zeros(NUM_STATES)
    
    # Địa chỉ ô "Trung tâm" (Trạng thái cân bằng nhất)
    # x=0, v=0, theta=0, omega=0
    S_ZERO = state_to_int(discretize(np.zeros(4)))
    
    rmse_td, rmse_mc = [], []
    v_start_td, v_start_mc = [], []
    
    active_states = np.nonzero(v_true > 0)[0]

    print("--- Bắt đầu CUỘC ĐUA (Comparing TD vs MC) ---")
    
    for ep in range(episodes):
        # Học tập TD(0) với Learning Rate Decay (giúp ổn định hơn)
        # alpha = initial_alpha / (1 + 0.00005 * ep) # Giảm chậm hơn 1 chút
        alpha = initial_alpha / (1 + 0.0001 * ep) 
        
        # 4.1 TD(0) Episode
        state, _ = env.reset()
        done = False
        while not done:
            s_idx = state_to_int(discretize(state))
            action = simple_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ns_idx = state_to_int(discretize(next_state))
            v_nx = 0 if done else v_td[ns_idx]
            v_td[s_idx] += alpha * (reward + gamma * v_nx - v_td[s_idx])
            state = next_state
            
        # 4.2 MC Episode
        state, _ = env.reset()
        traj = []
        done = False
        while not done:
            s_idx = state_to_int(discretize(state))
            action = simple_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            traj.append((s_idx, reward)); state = next_state
            
        G = 0; visited = set()
        for s_t, r_t in reversed(traj):
            G = r_t + gamma * G
            if s_t not in visited:
                v_mc[s_t] += initial_alpha * (G - v_mc[s_t])
                visited.add(s_t)
        
        # 4.3 Logging
        diff_td = v_td[active_states] - v_true[active_states]
        diff_mc = v_mc[active_states] - v_true[active_states]
        rmse_td.append(np.sqrt(np.mean(diff_td**2)))
        rmse_mc.append(np.sqrt(np.mean(diff_mc**2)))
        
        # Theo dõi giá trị của ô S_ZERO
        v_start_td.append(v_td[S_ZERO])
        v_start_mc.append(v_mc[S_ZERO])
        
        if (ep + 1) % 1000 == 0:
            print(f"  > Ep {ep+1} | RMSE_TD: {rmse_td[-1]:.2f} | V_Start_TD: {v_start_td[-1]:.2f}")
            
    return rmse_td, rmse_mc, v_start_td, v_start_mc, v_true[S_ZERO]

def moving_average(data, window=100):
    return np.convolve(data, np.ones(window)/window, mode='valid')

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    v_true = get_ground_truth(env)
    
    RACE_EP = 60000
    r_td, r_mc, v_td_h, v_mc_h, the_truth = run_stability_analysis(env, RACE_EP, 0.01, 0.99, v_true)
    
    # PHÂN TÍCH KẾT QUẢ VỚI 2 BIỂU ĐỒ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    window = 100
    
    # Biểu đồ 1: RMSE (Lỗi)
    ax1.plot(moving_average(r_td, window), label='TD(0) RMSE (Lỗi)', color='blue')
    ax1.plot(moving_average(r_mc, window), label='Monte Carlo RMSE (Lỗi)', color='red')
    ax1.set_title('BIỂU ĐỒ SOI LỖI (RMSE Reduction)')
    ax1.set_xlabel('Episodes'); ax1.set_ylabel('RMSE Error'); ax1.legend(); ax1.grid(True, alpha=0.3)
    
    # Biểu đồ 2: Convergence (Hội tụ giá trị)
    ax2.plot(moving_average(v_td_h, window), label='V_start TD(0)', color='blue', alpha=0.8)
    ax2.plot(moving_average(v_mc_h, window), label='V_start MC', color='red', alpha=0.8)
    ax2.axhline(y=the_truth, color='green', linestyle='--', label='THE TRUTH (Ground Truth)', linewidth=2)
    ax2.set_title('BIỂU ĐỒ HỘI TỤ GIÁ TRỊ V(s_start)')
    ax2.set_xlabel('Episodes'); ax2.set_ylabel('Value V'); ax2.legend(); ax2.grid(True, alpha=0.3)
    
    log_dir = "lab3/logs"; os.makedirs(log_dir, exist_ok=True)
    plt.savefig(os.path.join(log_dir, "stability_analysis.png"))
    print("--- Đã lưu phân tích chuyên sâu tại lab3/logs/stability_analysis.png ---")
    plt.show()
    env.close()
