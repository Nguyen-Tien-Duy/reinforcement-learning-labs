import numpy as np
import gymnasium as gym
import time
import sys
import os

# Parameters cho Discretization
BINS = [4, 4, 10, 10] # Vị trí, Vận tốc, Góc cột, Vận tốc góc
# Tổng số states: 4 * 4 * 10 * 10 = 1,600 states
MIN_VALS = np.array([-4.8, -5.0, -0.418, -5.0])
MAX_VALS = np.array([4.8, 5.0, 0.418, 5.0])

# Tổng số states: 1 * 1 * 12 * 12 = 144 states (Cực kỳ dày đặc mẫu!)
NUM_STATES = np.prod(BINS)
NUM_ACTIONS = 2 # 0: Trái, 1: Phải

def discretize(state):
    """
    Rời rạc hóa không gian liên tục thành các bins (discrete categories).
    Trạng thái được map vào khoảng [0, 1] và chia theo bins.
    """
    ratio = (state - MIN_VALS) / (MAX_VALS - MIN_VALS)
    indices = (ratio * BINS).astype(int)
    indices = np.clip(indices, 0, np.array(BINS) - 1)
    return tuple(indices)

def state_to_int(state_tuple):
    """ Chuyển tuple của các bins thành 1 số integer duy nhất (0 -> NUM_STATES-1) """
    idx = 0
    multiplier = 1
    for dim_size, val in zip(reversed(BINS), reversed(state_tuple)):
        idx += val * multiplier
        multiplier *= dim_size
    return idx

def collect_samples(env, num_episodes=100000):
    """ 
    Thu thập samples bằng Random Policy.
    Xây dựng bảng tần suất để ước lượng xác suất chuyển P(s'|s, a) và expected reward R(s, a).
    """
    print(f"Đang thu thập tập mẫu ({num_episodes} episodes)...", flush=True)
    transition_counts = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES))
    reward_sums = np.zeros((NUM_STATES, NUM_ACTIONS))
    state_action_counts = np.zeros((NUM_STATES, NUM_ACTIONS))
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        state_idx = state_to_int(discretize(state))
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_state_idx = state_to_int(discretize(next_state))
            
            transition_counts[state_idx, action, next_state_idx] += 1
            reward_sums[state_idx, action] += reward
            state_action_counts[state_idx, action] += 1
            
            state_idx = next_state_idx
            
    # Tính toán P và R
    P = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES))
    R = np.zeros((NUM_STATES, NUM_ACTIONS))
    
    for s in range(NUM_STATES):
        for a in range(NUM_ACTIONS):
            if state_action_counts[s, a] > 0:
                P[s, a] = transition_counts[s, a] / state_action_counts[s, a]
                R[s, a] = reward_sums[s, a] / state_action_counts[s, a]
            else:
                # Nếu không quan sát được, giả sử tự transition về chính nó kèm reward 0.
                P[s, a, s] = 1.0
                
    return P, R

def value_iteration(P, R, gamma=0.99, theta=1e-6):
    """ Thuật toán Value Iteration tìm ra Policy tối ưu """
    print("Bắt đầu chạy Value Iteration...")
    V = np.zeros(NUM_STATES)
    policy = np.zeros(NUM_STATES, dtype=int)
    iters = 0
    while True:
        delta = 0
        V_new = np.zeros(NUM_STATES)
        for s in range(NUM_STATES):
            Q_s = np.zeros(NUM_ACTIONS)
            for a in range(NUM_ACTIONS):
                # Theo phương trình toán học: Q(s, a) = R(s, a) + gamma * sum(P(s' | s, a) * V(s'))
                Q_s[a] = R[s, a] + gamma * np.sum(P[s, a] * V)
            
            best_val = np.max(Q_s)
            delta = max(delta, abs(best_val - V[s]))
            V_new[s] = best_val
            policy[s] = np.argmax(Q_s)
            
        V = V_new
        iters += 1
        
        if iters % 100 == 0:
            print(f"[VI] Vòng lặp: {iters:4d} | Độ lệch Max Delta: {delta:.6f}", flush=True)
            
        if delta < theta:
            break
    print(f"Value Iteration hội tụ sau {iters} vòng lặp.")
    return policy, V

def policy_evaluation(policy, P, R, gamma=0.99, theta=1e-6):
    """ Đánh giá (Evaluate) xem Value function cố định cho Policy hiện tại là bao nhiêu """
    V = np.zeros(NUM_STATES)
    eval_iters = 0
    while True:
        delta = 0
        V_new = np.zeros(NUM_STATES)
        for s in range(NUM_STATES):
            a = policy[s]
            v = R[s, a] + gamma * np.sum(P[s, a] * V)
            delta = max(delta, abs(v - V[s]))
            V_new[s] = v
        V = V_new
        eval_iters += 1
        if delta < theta:
            break
    return V, eval_iters

def policy_iteration(P, R, gamma=0.99, theta=1e-6, min_changes=13):
    """ Thuật toán Policy Iteration tìm ra Policy tối ưu """
    print("Bắt đầu chạy Policy Iteration...")
    policy = np.random.choice(NUM_ACTIONS, size=NUM_STATES)
    iters = 0
    while True:
        # Bước 1: Policy Evaluation
        V, eval_iters = policy_evaluation(policy, P, R, gamma, theta)
        
        # Bước 2: Policy Improvement
        policy_stable = True
        changes = 0
        for s in range(NUM_STATES):
            old_action = policy[s]
            Q_s = np.zeros(NUM_ACTIONS)
            for a in range(NUM_ACTIONS):
                Q_s[a] = R[s, a] + gamma * np.sum(P[s, a] * V)
            
            best_action = np.argmax(Q_s)
            policy[s] = best_action
            if old_action != best_action:
                policy_stable = False
                changes += 1
                
        iters += 1
        print(f"[PI] Vòng lớn: {iters:2d} | Vòng nhỏ Evaluation: {eval_iters:4d} | Điểm thay đổi Policy: {changes}", flush=True)
        if policy_stable or changes <= min_changes:
            print(f"Policy Iteration dừng sớm với {changes} changes còn lại.", flush=True)
            break
    print(f"Policy Iteration hội tụ sau {iters} vòng lặp.")
    return policy, V

def evaluate_policy(env, policy, num_episodes=100):
    """ Chạy agent trên environment sử dụng Policy đã train """
    total_reward = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            state_idx = state_to_int(discretize(state))
            if policy is None:
                action = env.action_space.sample()
            else:
                action = policy[state_idx]
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
    return total_reward / num_episodes

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "cartpole_dp_final.log")
    
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = log_file

    env = gym.make('CartPole-v1')
    
    # 1. Thu thập sample và estimate P, R
    print("================ BẮT ĐẦU CHẠY THUẬT TOÁN ================", flush=True)
    start_time = time.time()
    # Với 1,600 states, 60,000 episodes
    P, R = collect_samples(env, num_episodes=60000)
    print(f"-> Thời gian lấy mẫu, ước lượng Model P và R: {time.time() - start_time:.2f}s\n", flush=True)
    
    # 2. Chạy thuật toán Value Iteration
    start_time = time.time()
    vi_policy, vi_V = value_iteration(P, R)
    vi_time = time.time() - start_time
    
    # 3. Chạy thuật toán Policy Iteration
    print("")
    start_time = time.time()
    pi_policy, pi_V = policy_iteration(P, R)
    pi_time = time.time() - start_time
    
    # 4. Đánh giá và So sánh
    print("\n=== TỔNG KẾT & SO SÁNH ===")
    random_reward = evaluate_policy(env, policy=None)
    vi_reward = evaluate_policy(env, vi_policy)
    pi_reward = evaluate_policy(env, pi_policy)
    
    print(f" Tốc độ hội tụ thuật toán:")
    print(f"  - Value Iteration : {vi_time:.4f} giây")
    print(f"  - Policy Iteration: {pi_time:.4f} giây")
    
    print(f"\n Reward trung bình (trên 100 episodes):")
    print(f"  - Random Agent    : {random_reward:.2f}")
    print(f"  - Value Iteration : {vi_reward:.2f}")
    print(f"  - Policy Iteration: {pi_reward:.2f}")
    
    env.close()
    
    sys.stdout = sys.__stdout__
    log_file.close()