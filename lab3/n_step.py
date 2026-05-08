import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os

# 1. Cấu hình Rời rạc hóa (Discretization)
BINS = [4, 4, 10, 10] 
MIN_VALS = np.array([-4.8, -4.0, -0.418, -4.0])
MAX_VALS = np.array([4.8, 4.0, 0.418, 4.0])
NUM_STATES = np.prod(BINS)

def discretize(state):
    ratio = (state - MIN_VALS) / (MAX_VALS - MIN_VALS)
    indices = (ratio * BINS).astype(int)
    indices = np.clip(indices, 0, np.array(BINS) - 1)
    return tuple(indices)

def state_to_int(state_tuple):
    idx = 0; multiplier = 1
    for dim_size, val in zip(reversed(BINS), reversed(state_tuple)):
        idx += val * multiplier; multiplier *= dim_size
    return idx

def choose_action(state_idx, q_table, epsilon, env):
    if np.random.default_rng(42).random() < epsilon:
        return env.action_space.sample()
    return np.argmax(q_table[state_idx])

# 2. Thuật toán n-step Q-learning
def run_n_step_training(n, episodes, alpha=0.1, gamma=0.99, initial_epsilon=1.0, min_epsilon=0.01, decay=0.0005):
    env = gym.make("CartPole-v1")
    q_table = np.zeros((NUM_STATES, 2))
    reward_history = []
    epsilon = initial_epsilon
    
    print(f"--- Đang chạy n-step Q-learning với n={n} ---")
    
    for ep in range(episodes):
        state, _ = env.reset()
        state_idx = state_to_int(discretize(state))
        
        # Buffer lưu (s, a, r)
        buffer = deque(maxlen=n)
        total_reward = 0
        done = False
        
        while not done or len(buffer) > 0:
            if not done:
                action = choose_action(state_idx, q_table, epsilon, env)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                next_state_idx = state_to_int(discretize(next_state))
                buffer.append((state_idx, action, reward))
                
                state_idx = next_state_idx
                total_reward += reward
            
            # Cập nhật khi buffer đủ n bước hoặc episode kết thúc
            if len(buffer) == n or (done and len(buffer) > 0):
                # t là bước thời gian của kinh nghiệm đầu tiên trong buffer
                s_t, a_t, _ = buffer[0]
                
                # Tính n-step return G
                G = 0
                for i, (_, _, r_i) in enumerate(buffer):
                    G += (gamma ** i) * r_i
                
                # Cổng bootstrapping (max Q ở trạng thái s_{t+n})
                if not done:
                    G += (gamma ** len(buffer)) * np.max(q_table[state_idx])
                
                # Cập nhật Q(s_t, a_t)
                q_table[s_t, a_t] += alpha * (G - q_table[s_t, a_t])
                
                # Xóa kinh nghiệm đầu tiên khỏi buffer để trượt cửa sổ
                buffer.popleft()
        
        reward_history.append(total_reward)
        epsilon = max(min_epsilon, epsilon * np.exp(-decay * ep))
        
        if (ep + 1) % 2000 == 0:
            avg = np.mean(reward_history[-100:])
            print(f"  > Episode {ep+1}/{episodes} | Thưởng trung bình 100 eps: {avg:.2f}")
            
    env.close()
    return reward_history

def moving_average(data, window=100):
    return np.convolve(data, np.ones(window)/window, mode='valid')

if __name__ == "__main__":
    EPISODES = 10000
    
    # So sánh n = 1, 3, 5
    h1 = run_n_step_training(n=1, episodes=EPISODES)
    h3 = run_n_step_training(n=3, episodes=EPISODES)
    h5 = run_n_step_training(n=5, episodes=EPISODES)
    
    # Vẽ đồ thị
    plt.figure(figsize=(12, 7))
    window = 500
    plt.plot(moving_average(h1, window), label='n=1 (Regular Q-learning)', color='blue')
    plt.plot(moving_average(h3, window), label='n=3', color='green')
    plt.plot(moving_average(h5, window), label='n=5', color='red')
    
    plt.title('So sánh tốc độ học của n-step Q-learning trên CartPole')
    plt.xlabel('Episodes'); plt.ylabel('Thưởng trung bình (Moving Average)')
    plt.legend(); plt.grid(True, alpha=0.3)
    
    log_dir = "lab3/logs"; os.makedirs(log_dir, exist_ok=True)
    plt.savefig(os.path.join(log_dir, "n_step_comparison.png"))
    print("--- Đã lưu đồ thị so sánh tại lab3/logs/n_step_comparison.png ---")
    plt.show()
