import torch
import torch.nn as nn
import numpy as np
import random 
from Experience_Replay import ReplayMemory
from Q_network_relu import  DQNNet_relu 
import torch.nn.functional as F



class Double_DQNAgent:
    # this is the class of the agent 
    def __init__(self, device, state_size, action_size,
                    discount=0.99, 
                    eps_max=1.0, 
                    eps_min=0.01, 
                    eps_decay=0.995, 
                    memory_capacity=5000, 
                    lr=1e-4, 
                    train_mode=True):

        self.device = device

        
        self.epsilon = eps_max
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay

        
        self.discount = discount

        
        self.state_size = state_size
        self.action_size = action_size

        
        self.policy_net = DQNNet_relu(self.state_size, self.action_size, lr).to(self.device)
        self.target_net = DQNNet_relu(self.state_size, self.action_size, lr).to(self.device)
        self.best_net = DQNNet_relu(self.state_size, self.action_size, lr).to(self.device)
        self.best_net.eval()
        self.target_net.eval() 
        if not train_mode:
            self.policy_net.eval()

        
        self.memory = ReplayMemory(capacity=memory_capacity)


    def update_target_net(self):

        self.target_net.load_state_dict(self.policy_net.state_dict())


    def update_epsilon(self):
        
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)


    def select_action(self, state):

        if random.random() <= self.epsilon: 
            return random.randrange(self.action_size)

        if not torch.is_tensor(state):
            state = torch.tensor([state], dtype=torch.float32).to(self.device)

        
        with torch.no_grad():
            action = self.policy_net.forward(state)
        return torch.argmax(action).item() 


    def learn(self, batchsize):

        
        if len(self.memory) < batchsize:
            return
        states, actions, next_states, rewards, dones = self.memory.sample(batchsize, self.device)

        actions = torch.tensor(actions, dtype = torch.int64)
        q_pred = self.policy_net.forward(states).gather(1, actions.view(-1, 1)) 
        
        with torch.no_grad():
          acts = self.policy_net.forward(next_states)
        acts = torch.argmax(acts, dim = 1)
        acts = acts.view(-1,1)
        with torch.no_grad():
            q_target = self.target_net.forward(next_states).gather (1, acts)
        q_target[dones] = 0.0
        
        
        #q_target = self.target_net.forward(next_states).max(dim=1).values 
        #q_target[dones] = 0.0 
        
        y_j = rewards.view(-1,1) + (self.discount * q_target)
        y_j = y_j.view(-1, 1)
        
        
        self.policy_net.optimizer.zero_grad()
        loss = F.mse_loss(y_j, q_pred).mean()
        loss.backward()
        self.policy_net.optimizer.step()
        

    def save_model(self, filename):

        self.policy_net.save_model(filename)

    def load_model(self, filename):


        self.policy_net.load_model(filename=filename, device=self.device)
