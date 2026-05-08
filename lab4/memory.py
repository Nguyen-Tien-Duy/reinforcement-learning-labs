import random 
import numpy as np 
import torch
from collections import deque

class ReplayMemory:
    """This is class that store the experience of the agent"""
    def __init__(self, action_size, buffer_size, batch_size, device):
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.memory = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Randomly sample a batch of experience from the buffer"""
        experiences = random.sample(self.memory, k=self.batch_size)
        # Transform to tensor
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)

        # We can do the same thing with actions (long), reward (float), next_state (float), dones (uint8 or float)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None])).float().to(self.device)
        # Action is long tensor because it is discrete (and it's index)
        return (states, actions, rewards, next_states, dones)
        

    def __len__(self):
        """Return the number of experience in the buffer"""
        return len(self.memory)
