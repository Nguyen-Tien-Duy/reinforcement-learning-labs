import gymnasium as gym
import numpy as np 
import torch 
from agent import DQN 
import matplotlib.pyplot as plt 

def train():
    env = gym.make("LunarLander-v2")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n 
    agent = DQN(state_size, action_size, seed = 36)

    n_episodes = 500
    eps = 1.0
    eps_end = 0.01
    eps_decay = 0.995

    scores = []

    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        for t in range(1000):
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
        
        print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores[-100:]):.2f}", end="")
        if i_episode % 100 == 0:
            print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores[-100:]):.2f}")
            if np.mean(scores[-100:]) >= 200.0:
                print(f"\nEnvironment solved in {i_episode} episodes!")
                torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_dqn.pth')
                break

    # Plot results
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

if __name__ == "__main__":
    train()

    
