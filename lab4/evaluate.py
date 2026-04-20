import numpy as np 
import gymnasium as gym 
import torch
from model import QNetwork
from os import sys

def evaluate_agent(agent, env, n_episode=10):
    eval_scores = []
    for _ in range(n_episode):
        state, _ = env.reset()
        score = 0

        while True:
            action = agent.act(state, eps=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)

            score += reward
            state = next_state

            if terminated or truncated:
                break

        eval_scores.append(score)

    return np.mean(eval_scores)

def evaluate(n_episode=100):
    # env = gym.make("CartPole-v1", render_mode='human')
    env = gym.make("CartPole-v1")
    # Init enviroment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n 

    # We init modle and load weight
    model = QNetwork(state_size, action_size)
    checkpoint = torch.load("./lab4/checkpoint.pth", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Change mode model in to evaluation mode
    model.eval()

    for i in range(n_episode):
        state, _ = env.reset()
        score = 0
        done = False

        while not done:
            # Change state to tensor and put it into Network
            state_t = torch.from_numpy(state).float().unsqueeze(0)
            # Chosse the best action 
            with torch.no_grad():
                action_value = model(state_t)
            action = np.argmax(action_value.cpu().data.numpy())

            # Real action
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward

        print(f'Episode {i+1}: Score = {score}')

    env.close()

if __name__ == "__main__":
    evaluate(100)


