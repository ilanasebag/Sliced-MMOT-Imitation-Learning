import argparse
import pickle
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    
    def __init__(self, input_dim, output_dim, action_dim): 
        super(Net, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(self.input_dim, 100)
        self.fc2 = nn.Linear(100, self.action_dim)
        self.fc3 = nn.Linear (100, self.output_dim)
        
    def forward(self, state):
        state = F.tanh(self.fc1(state))
        a = self.fc2(state) - self.fc2(state).mean(1, keepdim = True)
        v = self.fc3(state)
        action_scores = a+v
        return action_scores
    
    
class Memory():

    data_pointer = 0
    isfull = False

    def __init__(self, capacity):
        self.memory = np.empty(capacity, dtype=object)
        self.capacity = capacity

    def update(self, transition):
        self.memory[self.data_pointer] = transition
        self.data_pointer += 1
        if self.data_pointer == self.capacity:
            self.data_pointer = 0
            self.isfull = True

    def sample(self, batch_size):
        return np.random.choice(self.memory, batch_size)


        
class Agent():

    def __init__(self, input_dim, output_dim, action_dim, environment):

        self.action_dim = action_dim 
        self.input_dim = input_dim
        self.output_dim = output_dim 
        self.environment = environment
        self.action_list = [(i * 4 - 2,) for i in range(self.action_dim)]
        #self.action_list = [(i  - 2,) for i in range(self.action_dim)]
        self.training_step = 0
        self.epsilon = 1
        self.eval_net, self.target_net = Net(self.input_dim, self.output_dim,
                                             self.action_dim).float(), Net(self.input_dim,
                                                                           self.output_dim,
                                                                           self.action_dim).float()
        self.memory = Memory(2000)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=1e-3)
        self.max_grad_norm = 0.5
        self.min_grad_norm = 0.1 
        self.GAMMA = 0.99

        
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        if np.random.random() < self.epsilon:
            action_index = np.random.randint(self.action_dim)
        else:
            probs = self.eval_net(state)
            action_index = probs.max(1)[1].item()
        return self.action_list[action_index], action_index
    

    def save_param(self):
        torch.save(self.eval_net.state_dict(), '/Users/ilanasebag/Documents/Thesis_code/RL_results/dqn_net_params_%s.pkl'%self.environment)

    def store_transition(self, transition):
        self.memory.update(transition)

    def update(self):
        self.training_step += 1

        transitions = self.memory.sample(32)
        s = torch.tensor([t.s for t in transitions], dtype=torch.float)
        a = torch.tensor([t.a for t in transitions], dtype=torch.long).view(-1, 1)
        r = torch.tensor([t.r for t in transitions], dtype=torch.float).view(-1, 1)
        s_ = torch.tensor([t.s_ for t in transitions], dtype=torch.float)

        # natural dqn
        # q_eval = self.eval_net(s).gather(1, a)
        # with torch.no_grad():
        #     q_target = r + args.gamma * self.target_net(s_).max(1, keepdim=True)[0]

        # double dqn
        with torch.no_grad():
            a_ = self.eval_net(s_).max(1, keepdim=True)[1]
            q_target = r + 0.99 * self.target_net(s_).gather(1, a_)
        q_eval = self.eval_net(s).gather(1, a)

        self.optimizer.zero_grad()
        loss = F.smooth_l1_loss(q_eval, q_target)
        loss.backward()
        nn.utils.clip_grad_norm_(self.eval_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if self.training_step % 200 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.epsilon = max(self.epsilon * self.GAMMA, 0.01)

        return q_eval.mean().item()
