import torch
import numpy as np 
import torch.nn.functional as F
import torch.optim as optim

from memory import ReplayMemory
from model import QNetwork

# We define Hyperparameters for DQN
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
LR = 5e-4
TAU = 1e-3
UPDATE_EVERY = 1

class DQN:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayMemory(action_size, BUFFER_SIZE, BATCH_SIZE, self.device)
        self.qnetwork_local = QNetwork(state_size, action_size).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.AdamW(self.qnetwork_local.parameters(), lr=LR)
        self.t_step = 0
        self.soft_update(self.qnetwork_local, self.qnetwork_target, 1.0)
    
    def act(self, state, eps=0.0):
        """Choose an action using epsilon-greedy policy"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if np.random.rand() > eps:
            with torch.no_grad():
                return np.argmax(self.qnetwork_local(state).detach().cpu().numpy())
        else:
            return np.random.choice(self.action_size)
    
    def step(self, state, action, reward, next_state, done):
        """Step method for updating the network"""
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        self.memory.add(state, action, reward, next_state, done)
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE)
                self.learn(experiences, GAMMA)
    
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            best_action = self.qnetwork_local(next_states).argmax(1).unsqueeze(1)
            # Get Q values for current states
            Q_target_next = self.qnetwork_target(next_states).gather(1, best_action)

        # Compute the target Q values
        Q_target = rewards + gamma * Q_target_next * (1 - dones)

        # Get Q values for actions taken
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Caculate Loss and Update
        loss = F.mse_loss(Q_expected, Q_target)

        # optimizer
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Update target network weights """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data) 

    

        
        
        
        