import numpy as np
import gymnasium as gym
import random
import logging
import os
import sys
import matplotlib.pyplot as plt

# Creaate log dir
log_dir = os.path.join(os.path.dirname(__file__), 'logs-sarsa')
os.makedirs(log_dir, exist_ok=True)
# create and config logging file for print instruction while training RL
log_file = os.path.join(log_dir, 'training.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Let's define hyperparameters
total_episodes = 80000 # Reduced slightly for faster 4-way comparison
learning_rate = 0.1
gamma = 0.99
max_epsilon = 1.0
min_epsilon = 0.001
decay_rate = 0.0001


def q_learning(is_slippery=True):
    # Khởi tạo epsilon nội bộ để mỗi lần chạy đều bắt đầu từ 1.0
    epsilon = max_epsilon

    logging.info(f"Khởi tạo môi trường FrozenLake-v1 (is_slippery={is_slippery})")
    env = gym.make('FrozenLake-v1', is_slippery=is_slippery)

    state_space = env.observation_space.n
    action_space = env.action_space.n
    
    # We need to create a Q-table to store the Q-values for each state-action pair
    q_table = np.zeros((state_space, action_space))

    # rewards list for each episode
    rewards_all_episodes = []

    # Let's start training
    logging.info('Start training process')

    for episode in range(total_episodes):
        # Reset environment
        state, info = env.reset()
        
        done = False
        truncated = False
        rewards_current_episode = 0

        while not (done or truncated):
            action = choose_action(epsilon, state, q_table, env)

            # Then take an action and observe the new state and reward
            new_state, reward, done, truncated, info = env.step(action)

            # So we updating Q-table
            # Q(s,a) = Q(s,a) + alpha * [R + gamma * max(Q(s',a')) - Q(s,a)]
            q_table[state,action] = q_table[state,action] + learning_rate * \
                (reward + gamma * np.max(q_table[new_state,:]) - q_table[state,action])
            
            state = new_state
            rewards_current_episode += reward

        rewards_all_episodes.append(rewards_current_episode)

        if (episode + 1) % 5000 == 0:
                avg_reward = sum(rewards_all_episodes[-5000:]) / 5000
                logging.info(f"Episode: {episode + 1}/{total_episodes} - Reward trung bình (5000 eps gần nhất): {avg_reward:.4f} - epsilon: {epsilon:.4f}")

        # decay epsilon over episodes
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        
    logging.info("Training process finished (Q-learning)")

    logging.info("Q-table:")
    logging.info(q_table)

    logging.info(f"Testing the trained agent (is_slippery={is_slippery})")
    env = gym.make('FrozenLake-v1', is_slippery=is_slippery)
    state, info = env.reset()
    done = False
    truncated = False
    rewards_current_episode = 0

    while not (done or truncated):
        action = np.argmax(q_table[state,:])
        new_state, reward, done, truncated, info = env.step(action)
        state = new_state
        rewards_current_episode += reward

    logging.info(f"Reward: {rewards_current_episode}")
    env.close()

    return rewards_all_episodes
        
def sarsa(is_slippery=True):
    # Khởi tạo epsilon nội bộ 
    epsilon = max_epsilon

    logging.info(f"Khởi tạo môi trường FrozenLake-v1 (is_slippery={is_slippery})")
    env = gym.make('FrozenLake-v1', is_slippery=is_slippery)

    # Get state space and action space
    state_space = env.observation_space.n
    action_space = env.action_space.n
    # init q table with all 0 
    q_table = np.zeros((state_space, action_space))

    # rewards list for each episode
    rewards_all_episodes = []

    # Let's start training
    logging.info('Start training process')

    for episode in range(total_episodes):
        # Reset environment
        state, info = env.reset()
        
        done = False
        truncated = False
        rewards_current_episode = 0

        action = choose_action(epsilon, state, q_table, env)

        while not (done or truncated):
            # Then take an action and observe the new state and reward
            new_state, reward, done, truncated, info = env.step(action)

            # So we updating Q-table, SARSA update rule
            new_action = choose_action(epsilon, new_state, q_table, env)
            # Q(s,a) = Q(s,a) + alpha * [R + gamma * Q(s',a') - Q(s,a)]
            q_table[state,action] = q_table[state,action] + learning_rate * \
                (reward + gamma * q_table[new_state,new_action] - q_table[state,action])
            
            state = new_state
            action = new_action
            rewards_current_episode += reward

        rewards_all_episodes.append(rewards_current_episode)

        if (episode + 1) % 5000 == 0:
                avg_reward = sum(rewards_all_episodes[-5000:]) / 5000
                logging.info(f"Episode: {episode + 1}/{total_episodes} - Reward trung bình (5000 eps gần nhất): {avg_reward:.4f} - epsilon: {epsilon:.4f}")

        # decay epsilon over episodes
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        
    logging.info("Training process finished (SARSA)")

    logging.info("Q-table:")
    logging.info(q_table)

    logging.info(f"Testing the trained agent (is_slippery={is_slippery})")
    env = gym.make('FrozenLake-v1', is_slippery=is_slippery)
    state, info = env.reset()
    done = False
    truncated = False
    rewards_current_episode = 0

    while not (done or truncated):
        action = np.argmax(q_table[state,:])
        new_state, reward, done, truncated, info = env.step(action)
        state = new_state
        rewards_current_episode += reward

    logging.info(f"Reward: {rewards_current_episode}")
    env.close()

    return rewards_all_episodes

    
def choose_action(epsilon ,state, q_table, env):
    exploration_rate = random.uniform(0,1)
    if exploration_rate > epsilon:
        #Exploitation
        action = np.argmax(q_table[state,:])
    else:
        #Exploration
        action = env.action_space.sample()
    return action

def moving_average(data, window_size=500):
    moving_avg = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        window_average = np.mean(window)
        moving_avg.append(window_average)
    return moving_avg
        
        
if __name__ == '__main__':
    window = 1000

    # 1. Chạy với is_slippery=True (Môi trường khó)
    logging.info("--- TRAINING SLIPPERY ENV ---")
    req_q_slip = q_learning(is_slippery=True)
    req_sarsa_slip = sarsa(is_slippery=True)

    # 2. Chạy với is_slippery=False (Môi trường dễ)
    logging.info("--- TRAINING NON-SLIPPERY ENV ---")
    req_q_non = q_learning(is_slippery=False)
    req_sarsa_non = sarsa(is_slippery=False)

    # Làm mịn dữ liệu
    q_slip_smooth = moving_average(req_q_slip, window)
    sarsa_slip_smooth = moving_average(req_sarsa_slip, window)
    q_non_smooth = moving_average(req_q_non, window)
    sarsa_non_smooth = moving_average(req_sarsa_non, window)

    # Vẽ biểu đồ so sánh cả 4 trường hợp
    plt.figure(figsize=(12, 8))
    
    # Nhóm Slippery (Đường nét đứt)
    plt.plot(q_slip_smooth, label='Q-Learning (Slippery)', color='blue', linestyle='--')
    plt.plot(sarsa_slip_smooth, label='SARSA (Slippery)', color='red', linestyle='--')
    
    # Nhóm Non-Slippery (Đường nét liền)
    plt.plot(q_non_smooth, label='Q-Learning (Non-Slippery)', color='blue', linewidth=2)
    plt.plot(sarsa_non_smooth, label='SARSA (Non-Slippery)', color='red', linewidth=2)
    
    plt.title('So sánh Learning Curve: Slippery vs Non-Slippery')
    plt.xlabel(f'Episodes (làm mịn qua {window} eps)')
    plt.ylabel('Phần thưởng trung bình')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Lưu và hiển thị
    plot_path = os.path.join(log_dir, "comparison_slippery.png")
    plt.savefig(plot_path)
    logging.info(f"Đã lưu biểu đồ so sánh tại: {plot_path}")
    plt.show()
    

    
    
