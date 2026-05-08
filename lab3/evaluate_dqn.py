import gymnasium as gym
import torch
import torch.nn as nn
import os
import time

# 1. Khai báo lại kiến trúc mạng
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def evaluate(episodes=10, render=False):
    # Khởi tạo môi trường
    if render:
        env = gym.make("CartPole-v1", render_mode="human")
    else:
        env = gym.make("CartPole-v1")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 2. Khởi tạo mạng Neural
    q_net = QNetwork(state_size, action_size)

    # 3. Nạp mô hình tốt nhất vào mạng
    model_path = os.path.join(os.path.dirname(__file__), 'logs', 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"Không tìm thấy file: {model_path}. Hãy chạy file train trước để tạo best_model.pth nhé!")
        return

    q_net.load_state_dict(torch.load(model_path, weights_only=True))
    
    # Ép mạng vào chế độ Evaluation 
    q_net.eval()

    print("=" * 50)
    print(" BẮT ĐẦU CHẾ ĐỘ ĐÁNH GIÁ (EVALUATION MODE)")
    print("=" * 50)

    total_rewards_list = []

    for e in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # 4. Agent trở thành Cỗ Máy Lạnh Lùng
            # Không khám phá ngẫu nhiên, chỉ 100% làm theo mạng Q
            with torch.no_grad():
                state_t = torch.FloatTensor(state)
                action = q_net(state_t).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
            
            # Nếu bật render, chờ một chút để mắt người kịp nhìn thấy
            if render:
                time.sleep(0.01)

        total_rewards_list.append(episode_reward)
        print(f"Episode {e + 1}/{episodes} - Reward: {episode_reward:.1f} (Tỉ lệ thắng: {'100%' if episode_reward == 500 else str(episode_reward/5) + '%'})")

    avg_reward = sum(total_rewards_list) / episodes
    print("=" * 50)
    print(f" Trung bình sảu {episodes} ván: {avg_reward:.1f}/500")
    if avg_reward == 500:
        print("KẾT LUẬN: Bạn đã tạo ra RL tốt")
    else:
        print(" Vẫn còn vấp ngã, cần train lâu hơn để có best_model xịn hơn.")
    print("=" * 50)

    env.close()

if __name__ == "__main__":
    # Test 10 ván liên tục xem nó có đạt 500/500 không
    evaluate(episodes=1000, render=False)
