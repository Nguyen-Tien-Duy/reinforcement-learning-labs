import gymnasium as gym 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import random
from collections import deque
import logging
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Create log directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)

# Configure logging
log_file = os.path.join(log_dir, 'training-deepRL.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)


# Define Neuron Network (Q_network)
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


#  Training Agent
def train():
    env = gym.make("CartPole-v1")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    q_net = QNetwork(state_size, action_size)
    target_net = QNetwork(state_size, action_size)
    target_net.load_state_dict(q_net.state_dict()) # Copy ban đầu
    target_net.eval() # Mạng target chỉ dùng để tính toán, không training trực tiếp

    optimizer = optim.AdamW(q_net.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20000) # T_max 
    memory = ReplayBuffer(100000)

    # Define Hyperparameters
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.02
    batch_size = 256
    episodes = 10000

    total_reward = []
    losses = []
    recent_rewards = deque(maxlen=50) # Lưu lịch sử 20 ván gần nhất
    best_avg_reward = 0 # Theo dõi phong độ trung bình cao nhất
    best_model_path = os.path.join(log_dir, 'best_model.pth')

    for e in range(episodes):
        state, _ = env.reset()

        episodes_reward = 0
        done = False

        while not done:
            # epsilon greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state)
                    action = q_net(state_t).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # Store in memory
            memory.push(state, action, reward, next_state, done)
            state = next_state
            episodes_reward += reward 

            if len(memory) > batch_size:
                batch = memory.sample(batch_size)
                # Transform batch to Tensor
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states_t = torch.FloatTensor(np.array(states))
                next_states_t = torch.FloatTensor(np.array(next_states))
                actions_t = torch.LongTensor(actions).view(-1, 1)
                rewards_t = torch.FloatTensor(rewards).view(-1, 1)
                dones_t = torch.FloatTensor(dones).view(-1, 1)
                
                # caculate current Q: Q(s, a)
                current_q = q_net(states_t).gather(1, actions_t)
                
                # Target: r + gamma * Q_target(s', argmax Q_online(s'))
                with torch.no_grad():
                    # Double DQN: Dùng q_net để chọn hành động tốt nhất cho s'
                    best_actions = q_net(next_states_t).argmax(1).view(-1, 1)
                    # Dùng target_net để định giá cho hành động đó
                    max_next_q = target_net(next_states_t).gather(1, best_actions)
                    target_q = rewards_t + (gamma * max_next_q * (1 - dones_t))
                
                # Loss and optim 
                loss = nn.MSELoss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0) # Tránh bùng nổ gradient
                optimizer.step()
                losses.append(loss.item())

                # Soft Update (Polyak Averaging) thay cho Hard Update
                tau = 0.005
                for target_param, local_param in zip(target_net.parameters(), q_net.parameters()):
                    target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        total_reward.append(episodes_reward)
        recent_rewards.append(episodes_reward)

        # Tính điểm trung bình của 10 ván gần nhất
        avg_reward = np.mean(recent_rewards)
        
        # Chiến thuật Checkpointing Xịn: Lưu lại Base trên phong độ trung bình
        if avg_reward > best_avg_reward and avg_reward >= 500:
            logging.warning(f"New best model found at episode {e} with average reward {avg_reward:.1f}")
            best_avg_reward = avg_reward
            torch.save(q_net.state_dict(), best_model_path)

        if e % 50 == 0:
            logging.info(f"Episode {e}, Reward: {episodes_reward}, Avg50: {avg_reward:.1f}, Epsilon: {epsilon:.2f}, LR: {optimizer.param_groups[0]['lr']}")

        scheduler.step() # Giảm dần Learning Rate theo hình cosin

    return total_reward, losses
                
rewards, losses = train()
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(rewards)
plt.title("Reward qua từng Episode")
plt.subplot(1, 2, 2)
plt.plot(losses)
plt.title("Loss (Sai số) qua từng bước học")
# Lưu ảnh với mốc thời gian để tránh ghi đè
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(log_dir, f"deepRL-reward_{timestamp}.svg")
plt.savefig(save_path)
plt.show()
