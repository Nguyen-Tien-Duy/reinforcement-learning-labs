import numpy as np
import gymnasium as gym
import random
import logging
import os
import sys

# Force utf-8 for stdout if running on Windows
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Create log directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)

# Configure logging
log_file = os.path.join(log_dir, 'training.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def main():
    logging.info("Khởi tạo môi trường FrozenLake-v1 (gymnasium)")
    # Môi trường stochastic với 16 state. is_slippery=True (mặc định) có trơn trượt
    env = gym.make('FrozenLake-v1', is_slippery=True)
               
    state_space = env.observation_space.n
    action_space = env.action_space.n
    logging.info(f"State space: {state_space}, Action space: {action_space}")

    # Bước 1 — Khởi tạo
    # Q-table = ma trận toàn số 0
    q_table = np.zeros((state_space, action_space))

    # Hyperparameters
    total_episodes = 10000        # Số episode cần train
    learning_rate = 0.8           # alpha
    gamma = 0.95                  # discount factor
    
    # Exploration parameters
    epsilon = 1.0                 # epsilon ban đầu (khám phá nhiều)
    max_epsilon = 1.0
    min_epsilon = 0.01          
    decay_rate = 0.001            # Tốc độ giảm epsilon

    # Theo dõi kết quả
    rewards_all_episodes = []

    logging.info("Bắt đầu huấn luyện...")
    # Bước 2 — Huấn luyện
    for episode in range(total_episodes):
        # 1. Reset môi trường
        state, info = env.reset()
        done = False
        truncated = False
        rewards_current_episode = 0

        # Lặp đến khi kết thúc episode
        while not (done or truncated):
            # 2. Chọn hành động theo epsilon-greedy
            exploration_rate_threshold = random.uniform(0, 1)
            if exploration_rate_threshold > epsilon:
                # Khai thác: chọn action Q lớn nhất
                action = np.argmax(q_table[state,:]) 
            else:
                # Khám phá: chọn action ngẫu nhiên
                action = env.action_space.sample()

            # 3. Step môi trường nhận (s', r)
            new_state, reward, done, truncated, info = env.step(action)

            # 4. Cập nhật Q-table (Off-policy TD Control: Q-learning)
            # Q(s,a) = Q(s,a) + alpha * [R + gamma * max(Q(s',a')) - Q(s,a)]
            q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
                learning_rate * (reward + gamma * np.max(q_table[new_state, :]))
            
            state = new_state
            rewards_current_episode += reward
        
        # 5. Giảm dần epsilon sau mỗi episode
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        
        rewards_all_episodes.append(rewards_current_episode)

        # Log kết quả định kỳ
        if (episode + 1) % 1000 == 0:
            avg_reward = sum(rewards_all_episodes[-1000:]) / 1000
            logging.info(f"Episode: {episode + 1}/{total_episodes} - Reward trung bình (1000 eps gần nhất): {avg_reward:.4f} - epsilon: {epsilon:.4f}")

    logging.info("Huấn luyện hoàn tất!")
    logging.info("\nQ-table hội tụ:")
    logging.info("\n" + str(q_table))
    
    # Đánh giá agent
    logging.info("--- Đánh giá Agent ---")
    eval_episodes = 100
    successful_episodes = 0
    env_eval = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
    
    for ep in range(eval_episodes):
        state, info = env_eval.reset()
        done = False
        truncated = False
        
        while not (done or truncated):
            # Chọn hành động tốt nhất từ Q-table (epsilon = 0)
            action = np.argmax(q_table[state,:])
            new_state, reward, done, truncated, info = env_eval.step(action)
            state = new_state
            
            if done:
                if reward == 1:
                    successful_episodes += 1

        
    logging.info(f"Đánh giá hoàn tất. Agent chạm đích thành công {successful_episodes}/{eval_episodes} episodes.")

    env.close()
    env_eval.close()

if __name__ == '__main__':
    main()
