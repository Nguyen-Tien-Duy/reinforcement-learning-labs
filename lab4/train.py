import gymnasium as gym
import numpy as np 
import torch 
from agent import DQN 
import matplotlib.pyplot as plt 
import os
from datetime import datetime
import logging
from evaluate import evaluate_agent

model_path = "lab4/dqn_cartpole_best.pth"

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


def train():
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n 
    agent = DQN(state_size, action_size, seed = 36)

    n_episodes = 1000
    eps = 1.0
    eps_end = 0.01
    eps_decay = 0.995

    scores = []

    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        while True:
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)

        if score >= 500:
            logging.info(f'Model gain 500 point! Revaluating though 10 episode')
            avg_eval = evaluate_agent(agent, env, n_episode=10)

            if avg_eval >= 500:
                logging.info(f'Good we have average eval score is {avg_eval}. Saving model ...')
                torch.save({
                    'episode': i_episode,
                    'model_state_dict': agent.qnetwork_local.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'score': avg_eval,
                }, "lab4/checkpoint.pth")
                logging.info(f"Model saved at {model_path}")
                break
            else:
                logging.info(f'Not quite stable! Average eval gain is {avg_eval}')
        
        print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores[-100:]):.2f}", end='')

    # Plot results
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(log_dir, f"deepRL-reward_{timestamp}.svg")
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    train()

    
