#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 21:36:05 2021

@author: ilanasebag
"""

# Imitation Learning notebook - Inverse Reinforcement Learning - Use rewards from sliced RL to imitate expert behavioural movements 

import gym
import numpy as np
import cvxpy as cp
import sys
import pylab
import matplotlib.pyplot as plt
import pandas as pd 
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
import random
import torch.nn as nn
import math
from itertools import count
from PIL import Image
import torch
import torch.optim 
import torchvision.transforms as transforms
from collections import namedtuple, deque
from torch import nn
from gym import make
import torch.optim as optim
from numpy import save
from tqdm.notebook import tqdm
import pickle

import pickle
from typing import Optional
import IPython
from IPython.display import set_matplotlib_formats; set_matplotlib_formats('svg')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from Code.utils import Net, Memory, Agent
from Code.sliced_wasserstein_rewards import *
from Code.plotting import preprocess_states, animate
from Code.helper_functions import *

## Load expert trajectories 

environment = 'Pendulum-v0'

t1 = 'multi_diff_lengths_excl_1'
traj1 = np.load('/Users/ilanasebag/Documents/Thesis_Code/RL_results/%s_exp_states_%s.npy'%(t1,environment))

t4 = 'multi_same_lengths_1'
traj4 = np.load('/Users/ilanasebag/Documents/Thesis_Code/RL_results/%s_exp_states_%s.npy'%(t4,environment))

t11 = 'multi_same_lengths_0_5'
traj11 = np.load('/Users/ilanasebag/Documents/Thesis_Code/RL_results/%s_exp_states_%s.npy'%(t11,environment))

t12 = 'multi_same_lengths_1_7'
traj12 = np.load('/Users/ilanasebag/Documents/Thesis_Code/RL_results/%s_exp_states_%s.npy'%(t12,environment))

t13 = 'multi_same_lengths_1_2'
traj13 = np.load('/Users/ilanasebag/Documents/Thesis_Code/RL_results/%s_exp_states_%s.npy'%(t13,environment))

t5 = 'simple_length_1'
traj5 = np.load('/Users/ilanasebag/Documents/Thesis_Code/RL_results/%s_exp_states_%s.npy'%(t5,environment))

t8 = 'simple_length_0_5'
traj8 = np.load('/Users/ilanasebag/Documents/Thesis_Code/RL_results/%s_exp_states_%s.npy'%(t8,environment))

t9 = 'simple_length_1_7'
traj9 = np.load('/Users/ilanasebag/Documents/Thesis_Code/RL_results/%s_exp_states_%s.npy'%(t9,environment))

t10 = 'simple_length_1_2'
traj10 = np.load('/Users/ilanasebag/Documents/Thesis_Code/RL_results/%s_exp_states_%s.npy'%(t10,environment))

## Imitation Learning Model

TrainingRecord = namedtuple('TrainingRecord', ['ep', 'reward'])
Transition = namedtuple('Transition', ['s', 'a', 'r', 's_'])

def main(environment, exp, seeds, simple = False, MMOT = False, wass_PWIL = False, PWIL = False):

    env = gym.make(environment)
    env.seed(seeds)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]
    action_dim = env.action_space.shape[0] * 5 #discretization  of the unique continuous action of the pendulum

    agent = Agent(input_dim, output_dim, action_dim, environment)

    training_records = []
    running_reward, running_q = -1000, 0
    

    for i_ep in range(800):

        rewards = []
        new_states = []
        old_states = []
        action_indexes = []

        score = 0

        #We fix the departure state 
        state = env.reset()
        env.env.state = np.array([np.pi/2, 0.5])
        env.env.last_u = None
        state = env.env._get_obs()
        
        #to make it more robust we have to use :
        #state = env.reset()
        
        for t in range(200):
            action, action_index = agent.select_action(state)
            state_, reward, done, _ = env.step(action)
            score += reward
            old_states.append(state)

            env.render()
            #agent.store_transition(Transition(state, action_index, (reward + 8) / 8, state_))
            state = state_
            if agent.memory.isfull:
                q = agent.update()
                running_q = 0.99 * running_q + 0.01 * q

            action_indexes.append(action_index)
            rewards.append(reward)
            new_states.append(state_)

        states_tens = [torch.tensor(elt) for elt in old_states] #agent rollout 
        states_tens = torch.stack(states_tens).float()

        
        if MMOT is True :
            rewards_multitask = rewarder_multi([states_tens, torch.tensor(exp[0]).float(), torch.tensor(exp[1]).float(), torch.tensor(exp[2]).float(), torch.tensor(exp[3]).float(), torch.tensor(exp[4]).float()], num_projections = 50)
        
        elif simple is True : 
            rewards_multitask = rewarder_multi([states_tens, torch.tensor(exp[0]).float()], num_projections = 50)
            
        elif wass_PWIL is True : 
            pwil_exp = torch.tensor(concatenate_and_sample(exp)).float()
            rewards_multitask = rewarder_multi([states_tens, pwil_exp], num_projections = 50)
            
        for t in range(200):
            rewards[t] = torch.exp(-5*rewards_multitask[t,0])
            agent.store_transition(Transition(old_states[t], action_indexes[t], rewards[t], new_states[t]))

        running_reward = running_reward * 0.9 + score * 0.1
        training_records.append(TrainingRecord(i_ep, running_reward))

       # print('Ep', i_ep, 'Average score:', running_reward, 'score of current env', score )

    env.close()
    
    return training_records
    
    
    

## Result plots over 10 seeds

### Experiment 1 : Unique expert trajectory - same length of the expert

#### Length = 1

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj5
    seeds = 31
    training_record_simple_l_1_seed1 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj5
    seeds = 32
    training_record_simple_l_1_seed2 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj5
    seeds = 33
    training_record_simple_l_1_seed3 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj5
    seeds = 34
    training_record_simple_l_1_seed4 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj5
    seeds = 35
    training_record_simple_l_1_seed5 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj5
    seeds = 36
    training_record_simple_l_1_seed6 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj5
    seeds = 37
    training_record_simple_l_1_seed7 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj5
    seeds = 38
    training_record_simple_l_1_seed8 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj5
    seeds = 39
    training_record_simple_l_1_seed9 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj5
    seeds = 40
    training_record_simple_l_1_seed10 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

l1_simple = [r.reward for r in training_record_simple_l_1_seed1]
l2_simple = [r.reward for r in training_record_simple_l_1_seed2]
l3_simple = [r.reward for r in training_record_simple_l_1_seed3]
l4_simple = [r.reward for r in training_record_simple_l_1_seed4]
l5_simple = [r.reward for r in training_record_simple_l_1_seed5]
l6_simple = [r.reward for r in training_record_simple_l_1_seed6]
l7_simple = [r.reward for r in training_record_simple_l_1_seed7]
l8_simple = [r.reward for r in training_record_simple_l_1_seed8]
l9_simple = [r.reward for r in training_record_simple_l_1_seed9]
l10_simple = [r.reward for r in training_record_simple_l_1_seed10]

multiple_lists = [l1_simple, l2_simple, l3_simple, l4_simple, l5_simple, l6_simple, l7_simple, l8_simple, l9_simple, l10_simple]
arrays = [np.array(x) for x in multiple_lists]
mean_lst = [np.mean(k) for k in zip(*arrays)]
std_lst = [np.std(g) for g in zip(*arrays)]
std_shade_pos = [sum(x) for x in zip(mean_lst, std_lst)]
std_shade_neg = [m - n for m,n in zip(mean_lst, std_lst)]

plt.plot([r.ep for r in training_record_simple_l_1_seed8], mean_lst, color = 'grey')
plt.fill_between( [r.ep for r in training_record_simple_l_1_seed8],std_shade_neg, std_shade_pos, color = 'lightgrey')
plt.xlabel('Episode')
plt.ylabel('Mean moving averaged reward')
plt.legend(['length = 1'])
plt.savefig("/Users/ilanasebag/Documents/Thesis_code/IL_results/simple_len_1_%s.png"%environment)

plt.show()

#### Length = 0.5

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj8
    seeds = 41
    training_record_simple_l_0_5_seed1 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj8
    seeds = 42
    training_record_simple_l_0_5_seed2 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj8
    seeds = 43
    training_record_simple_l_0_5_seed3 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj8
    seeds = 44
    training_record_simple_l_0_5_seed4 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj8
    seeds = 45
    training_record_simple_l_0_5_seed5 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj8
    seeds = 46
    training_record_simple_l_0_5_seed6 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj8
    seeds = 47
    training_record_simple_l_0_5_seed7 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj8
    seeds = 48
    training_record_simple_l_0_5_seed8 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj8
    seeds = 49
    training_record_simple_l_0_5_seed9 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj8
    seeds = 50
    training_record_simple_l_0_5_seed10 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

l1_simple = [r.reward for r in training_record_simple_l_0_5_seed1]
l2_simple = [r.reward for r in training_record_simple_l_0_5_seed2]
l3_simple = [r.reward for r in training_record_simple_l_0_5_seed3]
l4_simple = [r.reward for r in training_record_simple_l_0_5_seed4]
l5_simple = [r.reward for r in training_record_simple_l_0_5_seed5]
l6_simple = [r.reward for r in training_record_simple_l_0_5_seed6]
l7_simple = [r.reward for r in training_record_simple_l_0_5_seed7]
l8_simple = [r.reward for r in training_record_simple_l_0_5_seed8]
l9_simple = [r.reward for r in training_record_simple_l_0_5_seed9]
l10_simple = [r.reward for r in training_record_simple_l_0_5_seed10]

multiple_lists = [l1_simple, l2_simple, l3_simple, l4_simple, l5_simple, l6_simple, l7_simple, l8_simple, l9_simple, l10_simple]
arrays = [np.array(x) for x in multiple_lists]
mean_lst = [np.mean(k) for k in zip(*arrays)]
std_lst = [np.std(g) for g in zip(*arrays)]
std_shade_pos = [sum(x) for x in zip(mean_lst, std_lst)]
std_shade_neg = [m - n for m,n in zip(mean_lst, std_lst)]

plt.plot([r.ep for r in training_record_simple_l_0_5_seed3], mean_lst, color = 'grey')
plt.fill_between( [r.ep for r in training_record_simple_l_0_5_seed3],std_shade_neg, std_shade_pos, color = 'lightgrey')
plt.xlabel('Episode')
plt.ylabel('Mean moving averaged reward')
plt.legend(['length = 0.5'])
plt.savefig("/Users/ilanasebag/Documents/Thesis_code/IL_results/simple_len_0_5_%s.png"%environment)

plt.show()

#### Length = 1.7

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj9
    seeds = 51
    training_record_simple_l_1_7_seed1 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj9
    seeds = 52
    training_record_simple_l_1_7_seed2 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj9
    seeds = 53
    training_record_simple_l_1_7_seed3 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj9
    seeds = 54
    training_record_simple_l_1_7_seed4 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj9
    seeds = 55
    training_record_simple_l_1_7_seed5 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj9
    seeds = 56
    training_record_simple_l_1_7_seed6 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj9
    seeds = 57
    training_record_simple_l_1_7_seed7 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj9
    seeds = 58
    training_record_simple_l_1_7_seed8 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj9
    seeds = 59
    training_record_simple_l_1_7_seed9 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj9
    seeds = 60
    training_record_simple_l_1_7_seed10 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

l1_simple = [r.reward for r in training_record_simple_l_1_7_seed1]
l2_simple = [r.reward for r in training_record_simple_l_1_7_seed2]
l3_simple = [r.reward for r in training_record_simple_l_1_7_seed3]
l4_simple = [r.reward for r in training_record_simple_l_1_7_seed4]
l5_simple = [r.reward for r in training_record_simple_l_1_7_seed5]
l6_simple = [r.reward for r in training_record_simple_l_1_7_seed6]
l7_simple = [r.reward for r in training_record_simple_l_1_7_seed7]
l8_simple = [r.reward for r in training_record_simple_l_1_7_seed8]
l9_simple = [r.reward for r in training_record_simple_l_1_7_seed9]
l10_simple = [r.reward for r in training_record_simple_l_1_7_seed10]

multiple_lists = [l1_simple, l2_simple, l3_simple, l4_simple, l5_simple, l6_simple, l7_simple, l8_simple, l9_simple, l10_simple]
arrays = [np.array(x) for x in multiple_lists]
mean_lst = [np.mean(k) for k in zip(*arrays)]
std_lst = [np.std(g) for g in zip(*arrays)]
std_shade_pos = [sum(x) for x in zip(mean_lst, std_lst)]
std_shade_neg = [m - n for m,n in zip(mean_lst, std_lst)]

plt.plot([r.ep for r in training_record_simple_l_1_7_seed10], mean_lst, color = 'grey')
plt.fill_between( [r.ep for r in training_record_simple_l_1_7_seed10],std_shade_neg, std_shade_pos, color = 'lightgrey')
plt.xlabel('Episode')
plt.ylabel('Mean moving averaged reward')
plt.legend(['length = 1.7'])
plt.savefig("/Users/ilanasebag/Documents/Thesis_code/IL_results/simple_len_1_7_%s.png"%environment)

plt.show()

#### Length = 1.2

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj10
    seeds = 61
    training_record_simple_l_1_2_seed1 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj10
    seeds = 62
    training_record_simple_l_1_2_seed2 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj10
    seeds = 63
    training_record_simple_l_1_2_seed3 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj10
    seeds = 64
    training_record_simple_l_1_2_seed4 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj10
    seeds = 65
    training_record_simple_l_1_2_seed5 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj10
    seeds = 66
    training_record_simple_l_1_2_seed6 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj10
    seeds = 67
    training_record_simple_l_1_2_seed7 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj10
    seeds = 68
    training_record_simple_l_1_2_seed8 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj10
    seeds = 69
    training_record_simple_l_1_2_seed9 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj10
    seeds = 70
    training_record_simple_l_1_2_seed10 = main(environment, exp, seeds, simple = True, MMOT = False, wass_PWIL = False, PWIL = False)

l1_simple = [r.reward for r in training_record_simple_l_1_2_seed1]
l2_simple = [r.reward for r in training_record_simple_l_1_2_seed2]
l3_simple = [r.reward for r in training_record_simple_l_1_2_seed3]
l4_simple = [r.reward for r in training_record_simple_l_1_2_seed4]
l5_simple = [r.reward for r in training_record_simple_l_1_2_seed5]
l6_simple = [r.reward for r in training_record_simple_l_1_2_seed6]
l7_simple = [r.reward for r in training_record_simple_l_1_2_seed7]
l8_simple = [r.reward for r in training_record_simple_l_1_2_seed8]
l9_simple = [r.reward for r in training_record_simple_l_1_2_seed9]
l10_simple = [r.reward for r in training_record_simple_l_1_2_seed10]

multiple_lists = [l1_simple, l2_simple, l3_simple, l4_simple, l5_simple, l6_simple, l7_simple, l8_simple, l9_simple, l10_simple]
arrays = [np.array(x) for x in multiple_lists]
mean_lst = [np.mean(k) for k in zip(*arrays)]
std_lst = [np.std(g) for g in zip(*arrays)]
std_shade_pos = [sum(x) for x in zip(mean_lst, std_lst)]
std_shade_neg = [m - n for m,n in zip(mean_lst, std_lst)]

plt.plot([r.ep for r in training_record_simple_l_1_2_seed6], mean_lst, color = 'grey')
plt.fill_between( [r.ep for r in training_record_simple_l_1_2_seed6],std_shade_neg, std_shade_pos, color = 'lightgrey')
plt.xlabel('Episode')
plt.ylabel('Mean moving averaged reward')
plt.legend(['length = 1.2'])
plt.savefig("/Users/ilanasebag/Documents/Thesis_code/IL_results/simple_len_1_2_%s.png"%environment)

plt.show()

### Experiment 2 : 5 expert trajectories with unique length

#### Length = 1

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj4
    seeds = 71
    training_record_multi_l_1_seed1 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj4
    seeds = 72
    training_record_multi_l_1_seed2 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj4
    seeds = 73
    training_record_multi_l_1_seed3 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj4
    seeds = 74
    training_record_multi_l_1_seed4 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj4
    seeds = 75
    training_record_multi_l_1_seed5 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj4
    seeds = 76
    training_record_multi_l_1_seed6 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj4
    seeds = 77
    training_record_multi_l_1_seed7 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj4
    seeds = 78
    training_record_multi_l_1_seed8 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj4
    seeds = 79
    training_record_multi_l_1_seed9 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj4
    seeds = 80
    training_record_multi_l_1_seed10 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

l1_simple = [r.reward for r in training_record_multi_l_1_seed1]
l2_simple = [r.reward for r in training_record_multi_l_1_seed2]
l3_simple = [r.reward for r in training_record_multi_l_1_seed3]
l4_simple = [r.reward for r in training_record_multi_l_1_seed4]
l5_simple = [r.reward for r in training_record_multi_l_1_seed5]
l6_simple = [r.reward for r in training_record_multi_l_1_seed6]
l7_simple = [r.reward for r in training_record_multi_l_1_seed7]
l8_simple = [r.reward for r in training_record_multi_l_1_seed8]
l9_simple = [r.reward for r in training_record_multi_l_1_seed9]
l10_simple = [r.reward for r in training_record_multi_l_1_seed10]

multiple_lists = [l1_simple, l2_simple, l3_simple, l4_simple, l5_simple, l6_simple, l7_simple, l8_simple, l9_simple, l10_simple]
arrays = [np.array(x) for x in multiple_lists]
mean_lst = [np.mean(k) for k in zip(*arrays)]
std_lst = [np.std(g) for g in zip(*arrays)]
std_shade_pos = [sum(x) for x in zip(mean_lst, std_lst)]
std_shade_neg = [m - n for m,n in zip(mean_lst, std_lst)]

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj4
    seeds = 711
    training_record_multi_l_1_seed1 = main(environment, exp, seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj4
    seeds = 721
    training_record_multi_l_1_seed2 = main(environment, exp, seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj4
    seeds = 731
    training_record_multi_l_1_seed3 = main(environment, exp, seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj4
    seeds = 741
    training_record_multi_l_1_seed4 = main(environment, exp, seeds, simple = False,MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj4
    seeds = 751
    training_record_multi_l_1_seed5 = main(environment, exp, seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj4
    seeds = 761
    training_record_multi_l_1_seed6 = main(environment, exp, seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj4
    seeds = 771
    training_record_multi_l_1_seed7 = main(environment, exp, seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj4
    seeds = 781
    training_record_multi_l_1_seed8 = main(environment, exp, seeds, simple = False,MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj4
    seeds = 791
    training_record_multi_l_1_seed9 = main(environment, exp, seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj4
    seeds = 801
    training_record_multi_l_1_seed10 = main(environment, exp, seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

l1_simple = [r.reward for r in training_record_multi_l_1_seed1]
l2_simple = [r.reward for r in training_record_multi_l_1_seed2]
l3_simple = [r.reward for r in training_record_multi_l_1_seed3]
l4_simple = [r.reward for r in training_record_multi_l_1_seed4]
l5_simple = [r.reward for r in training_record_multi_l_1_seed5]
l6_simple = [r.reward for r in training_record_multi_l_1_seed6]
l7_simple = [r.reward for r in training_record_multi_l_1_seed7]
l8_simple = [r.reward for r in training_record_multi_l_1_seed8]
l9_simple = [r.reward for r in training_record_multi_l_1_seed9]
l10_simple = [r.reward for r in training_record_multi_l_1_seed10]

multiple_lists = [l1_simple, l2_simple, l3_simple, l4_simple, l5_simple, l6_simple, l7_simple, l8_simple, l9_simple, l10_simple]
arrays = [np.array(x) for x in multiple_lists]
_mean_lst = [np.mean(k) for k in zip(*arrays)]
_std_lst = [np.std(g) for g in zip(*arrays)]
_std_shade_pos = [sum(x) for x in zip(_mean_lst, _std_lst)]
_std_shade_neg = [m - n for m,n in zip(_mean_lst, _std_lst)]

plt.plot([r.ep for r in training_record_multi_l_1_seed1], mean_lst, color = 'orange')
plt.fill_between( [r.ep for r in training_record_multi_l_1_seed1],std_shade_neg, std_shade_pos, color = 'mocassin')

plt.plot([r.ep for r in training_record_multi_l_1_seed1], _mean_lst, color = 'royalblue')
plt.fill_between( [r.ep for r in training_record_multi_l_1_seed1],_std_shade_neg, _std_shade_pos, color = 'lightsteelblue')

plt.xlabel('Episode')
plt.ylabel('Mean moving averaged reward')
plt.legend(['SMMOT, SWPWIL'])
plt.title('Length = 1')
plt.savefig("/Users/ilanasebag/Documents/Thesis_code/IL_results/multi_unique_len_1_%s.png"%environment)
plt.show()


#### Length = 0.5

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj11
    seeds = 81
    training_record_multi_l_0_5_seed1 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj11
    seeds = 82
    training_record_multi_l_0_5_seed2 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj11
    seeds = 83
    training_record_multi_l_0_5_seed3= main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj11
    seeds = 84
    training_record_multi_l_0_5_seed4 = main(environment, exp, seeds,simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj11
    seeds = 85
    training_record_multi_l_0_5_seed5 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj11
    seeds = 86
    training_record_multi_l_0_5_seed6 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj11
    seeds = 87
    training_record_multi_l_0_5_seed7 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj11
    seeds = 88
    training_record_multi_l_0_5_seed8 = main(environment, exp, seeds,simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj11
    seeds = 89
    training_record_multi_l_0_5_seed9 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj11
    seeds = 90
    training_record_multi_l_0_5_seed10 = main(environment, exp, seeds,simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

l1_simple = [r.reward for r in training_record_multi_l_0_5_seed1]
l2_simple = [r.reward for r in training_record_multi_l_0_5_seed2]
l3_simple = [r.reward for r in training_record_multi_l_0_5_seed3]
l4_simple = [r.reward for r in training_record_multi_l_0_5_seed4]
l5_simple = [r.reward for r in training_record_multi_l_0_5_seed5]
l6_simple = [r.reward for r in training_record_multi_l_0_5_seed6]
l7_simple = [r.reward for r in training_record_multi_l_0_5_seed7]
l8_simple = [r.reward for r in training_record_multi_l_0_5_seed8]
l9_simple = [r.reward for r in training_record_multi_l_0_5_seed9]
l10_simple = [r.reward for r in training_record_multi_l_0_5_seed10]

multiple_lists = [l1_simple, l2_simple, l3_simple, l4_simple, l5_simple, l6_simple, l7_simple, l8_simple, l9_simple, l10_simple]
arrays = [np.array(x) for x in multiple_lists]
mean_lst = [np.mean(k) for k in zip(*arrays)]
std_lst = [np.std(g) for g in zip(*arrays)]
std_shade_pos = [sum(x) for x in zip(mean_lst, std_lst)]
std_shade_neg = [m - n for m,n in zip(mean_lst, std_lst)]

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj11
    seeds = 811
    training_record_multi_l_0_5_seed1 = main(environment, exp, seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj11
    seeds = 821
    training_record_multi_l_0_5_seed2 = main(environment, exp, seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj11
    seeds = 831
    training_record_multi_l_0_5_seed3= main(environment, exp, seeds, simple = False,  MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj11
    seeds = 841
    training_record_multi_l_0_5_seed4 = main(environment, exp, seeds,simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj11
    seeds = 851
    training_record_multi_l_0_5_seed5 = main(environment, exp, seeds, simple = False,  MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj11
    seeds = 861
    training_record_multi_l_0_5_seed6 = main(environment, exp, seeds, simple = False,  MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj11
    seeds = 871
    training_record_multi_l_0_5_seed7 = main(environment, exp, seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj11
    seeds = 881
    training_record_multi_l_0_5_seed8 = main(environment, exp, seeds,simple = False,  MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj11
    seeds = 891
    training_record_multi_l_0_5_seed9 = main(environment, exp, seeds, simple = False,  MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj11
    seeds = 901
    training_record_multi_l_0_5_seed10 = main(environment, exp, seeds,simple = False,  MMOT = False, wass_PWIL = True, PWIL = False)

l1_simple = [r.reward for r in training_record_multi_l_0_5_seed1]
l2_simple = [r.reward for r in training_record_multi_l_0_5_seed2]
l3_simple = [r.reward for r in training_record_multi_l_0_5_seed3]
l4_simple = [r.reward for r in training_record_multi_l_0_5_seed4]
l5_simple = [r.reward for r in training_record_multi_l_0_5_seed5]
l6_simple = [r.reward for r in training_record_multi_l_0_5_seed6]
l7_simple = [r.reward for r in training_record_multi_l_0_5_seed7]
l8_simple = [r.reward for r in training_record_multi_l_0_5_seed8]
l9_simple = [r.reward for r in training_record_multi_l_0_5_seed9]
l10_simple = [r.reward for r in training_record_multi_l_0_5_seed10]

multiple_lists = [l1_simple, l2_simple, l3_simple, l4_simple, l5_simple, l6_simple, l7_simple, l8_simple, l9_simple, l10_simple]
arrays = [np.array(x) for x in multiple_lists]
_mean_lst = [np.mean(k) for k in zip(*arrays)]
_std_lst = [np.std(g) for g in zip(*arrays)]
_std_shade_pos = [sum(x) for x in zip(_mean_lst, _std_lst)]
_std_shade_neg = [m - n for m,n in zip(_mean_lst, _std_lst)]

plt.plot([r.ep for r in training_record_multi_l_0_5_seed10], mean_lst, color = 'orange')
plt.fill_between( [r.ep for r in training_record_multi_l_0_5_seed10],std_shade_neg, std_shade_pos, color = 'mocassin')

plt.plot([r.ep for r in training_record_multi_l_0_5_seed10], _mean_lst, color = 'royalblue')
plt.fill_between( [r.ep for r in training_record_multi_l_0_5_seed10],_std_shade_neg, _std_shade_pos, color = 'lightsteelblue')

plt.xlabel('Episode')
plt.ylabel('Mean moving averaged reward')
plt.title('Length = 0.5')
plt.legend(['SMMOT', 'SWPWIL'])
plt.savefig("/Users/ilanasebag/Documents/Thesis_code/IL_results/multi_unique_len_0_5_%s.png"%environment)
plt.show()

#### Length = 1.7

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj12
    seeds = 91
    training_record_multi_l_1_7_seed1 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj12
    seeds = 92
    training_record_multi_l_1_7_seed2= main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj12
    seeds = 93
    training_record_multi_l_1_7_seed3 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj12
    seeds = 94
    training_record_multi_l_1_7_seed4 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj12
    seeds = 95
    training_record_multi_l_1_7_seed5 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj12
    seeds = 96
    training_record_multi_l_1_7_seed6 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj12
    seeds = 97
    training_record_multi_l_1_7_seed7 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj12
    seeds = 98
    training_record_multi_l_1_7_seed8 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj12
    seeds = 99
    training_record_multi_l_1_7_seed9 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj12
    seeds = 100
    training_record_multi_l_1_7_seed10 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

l1_simple = [r.reward for r in training_record_multi_l_1_7_seed1]
l2_simple = [r.reward for r in training_record_multi_l_1_7_seed2]
l3_simple = [r.reward for r in training_record_multi_l_1_7_seed3]
l4_simple = [r.reward for r in training_record_multi_l_1_7_seed4]
l5_simple = [r.reward for r in training_record_multi_l_1_7_seed5]
l6_simple = [r.reward for r in training_record_multi_l_1_7_seed6]
l7_simple = [r.reward for r in training_record_multi_l_1_7_seed7]
l8_simple = [r.reward for r in training_record_multi_l_1_7_seed8]
l9_simple = [r.reward for r in training_record_multi_l_1_7_seed9]
l10_simple = [r.reward for r in training_record_multi_l_1_7_seed10]

multiple_lists = [l1_simple, l2_simple, l3_simple, l4_simple, l5_simple, l6_simple, l7_simple, l8_simple, l9_simple, l10_simple]
arrays = [np.array(x) for x in multiple_lists]
mean_lst = [np.mean(k) for k in zip(*arrays)]
std_lst = [np.std(g) for g in zip(*arrays)]
std_shade_pos = [sum(x) for x in zip(mean_lst, std_lst)]
std_shade_neg = [m - n for m,n in zip(mean_lst, std_lst)]

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj12
    seeds = 911
    training_record_multi_l_1_7_seed1 = main(environment, exp, seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj12
    seeds = 921
    training_record_multi_l_1_7_seed2= main(environment, exp, seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj12
    seeds = 931
    training_record_multi_l_1_7_seed3 = main(environment, exp, seeds, simple = False,MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj12
    seeds = 941
    training_record_multi_l_1_7_seed4 = main(environment, exp, seeds, simple = False,MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj12
    seeds = 951
    training_record_multi_l_1_7_seed5 = main(environment, exp, seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj12
    seeds = 961
    training_record_multi_l_1_7_seed6 = main(environment, exp, seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj12
    seeds = 971
    training_record_multi_l_1_7_seed7 = main(environment, exp, seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj12
    seeds = 981
    training_record_multi_l_1_7_seed8 = main(environment, exp, seeds, simple = False,MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj12
    seeds = 991
    training_record_multi_l_1_7_seed9 = main(environment, exp, seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj12
    seeds = 1001
    training_record_multi_l_1_7_seed10 = main(environment, exp, seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

l1_simple = [r.reward for r in training_record_multi_l_1_7_seed1]
l2_simple = [r.reward for r in training_record_multi_l_1_7_seed2]
l3_simple = [r.reward for r in training_record_multi_l_1_7_seed3]
l4_simple = [r.reward for r in training_record_multi_l_1_7_seed4]
l5_simple = [r.reward for r in training_record_multi_l_1_7_seed5]
l6_simple = [r.reward for r in training_record_multi_l_1_7_seed6]
l7_simple = [r.reward for r in training_record_multi_l_1_7_seed7]
l8_simple = [r.reward for r in training_record_multi_l_1_7_seed8]
l9_simple = [r.reward for r in training_record_multi_l_1_7_seed9]
l10_simple = [r.reward for r in training_record_multi_l_1_7_seed10]

multiple_lists = [l1_simple, l2_simple, l3_simple, l4_simple, l5_simple, l6_simple, l7_simple, l8_simple, l9_simple, l10_simple]
arrays = [np.array(x) for x in multiple_lists]
_mean_lst = [np.mean(k) for k in zip(*arrays)]
_std_lst = [np.std(g) for g in zip(*arrays)]
_std_shade_pos = [sum(x) for x in zip(_mean_lst, _std_lst)]
_std_shade_neg = [m - n for m,n in zip(_mean_lst, _std_lst)]

plt.plot([r.ep for r in training_record_multi_l_1_7_seed10], mean_lst, color = 'orange')
plt.fill_between( [r.ep for r in training_record_multi_l_1_7_seed10],std_shade_neg, std_shade_pos, color = 'mocassin')

plt.plot([r.ep for r in training_record_multi_l_1_7_seed10], _mean_lst, color = 'royalblue')
plt.fill_between( [r.ep for r in training_record_multi_l_1_7_seed10],_std_shade_neg, _std_shade_pos, color = 'lightsteelblue')
plt.xlabel('Episode')
plt.ylabel('Mean moving averaged reward')
plt.legend(['SMMOT', 'SWPWIL'])
plt.title('Length = 1.7')
plt.savefig("/Users/ilanasebag/Documents/Thesis_code/IL_results/multi_unique_len_1_7_%s.png"%environment)
plt.show()

#### Length = 1.2

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj13
    seeds = 191
    training_record_multi_l_2_seed1 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj13
    seeds = 192
    training_record_multi_l_1_2_seed2= main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj13
    seeds = 193
    training_record_multi_l_1_2_seed3 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj13
    seeds = 194
    training_record_multi_l_1_2_seed4 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj13
    seeds = 195
    training_record_multi_l_1_2_seed5 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj13
    seeds = 196
    training_record_multi_l_1_2_seed6 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj13
    seeds = 197
    training_record_multi_l_1_2_seed7 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj13
    seeds = 198
    training_record_multi_l_1_2_seed8 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj13
    seeds = 199
    training_record_multi_l_1_2_seed9 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj13
    seeds = 1100
    training_record_multi_l_1_2_seed10 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

l1_simple = [r.reward for r in training_record_multi_l_1_2_seed1]
l2_simple = [r.reward for r in training_record_multi_l_1_2_seed2]
l3_simple = [r.reward for r in training_record_multi_l_1_2_seed3]
l4_simple = [r.reward for r in training_record_multi_l_1_2_seed4]
l5_simple = [r.reward for r in training_record_multi_l_1_2_seed5]
l6_simple = [r.reward for r in training_record_multi_l_1_2_seed6]
l7_simple = [r.reward for r in training_record_multi_l_1_2_seed7]
l8_simple = [r.reward for r in training_record_multi_l_1_2_seed8]
l9_simple = [r.reward for r in training_record_multi_l_1_2_seed9]
l10_simple = [r.reward for r in training_record_multi_l_1_2_seed10]

multiple_lists = [l1_simple, l2_simple, l3_simple, l4_simple, l5_simple, l6_simple, l7_simple, l8_simple, l9_simple, l10_simple]
arrays = [np.array(x) for x in multiple_lists]
mean_lst = [np.mean(k) for k in zip(*arrays)]
std_lst = [np.std(g) for g in zip(*arrays)]
std_shade_pos = [sum(x) for x in zip(mean_lst, std_lst)]
std_shade_neg = [m - n for m,n in zip(mean_lst, std_lst)]

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj13
    seeds = 1911
    training_record_multi_l_1_2_seed1 = main(environment, exp, seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj13
    seeds = 1921
    training_record_multi_l_1_2_seed2= main(environment, exp, seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj13
    seeds = 1931
    training_record_multi_l_1_2_seed3 = main(environment, exp, seeds, simple = False,MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj13
    seeds = 1941
    training_record_multi_l_1_2_seed4 = main(environment, exp, seeds, simple = False,MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj13
    seeds = 1951
    training_record_multi_l_1_2_seed5 = main(environment, exp, seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj13
    seeds = 1961
    training_record_multi_l_1_2_seed6 = main(environment, exp, seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj13
    seeds = 1971
    training_record_multi_l_1_2_seed7 = main(environment, exp, seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj13
    seeds = 1981
    training_record_multi_l_1_2_seed8 = main(environment, exp, seeds, simple = False,MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj13
    seeds = 1991
    training_record_multi_l_1_2_seed9 = main(environment, exp, seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj13
    seeds = 11001
    training_record_multi_l_1_2_seed10 = main(environment, exp, seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

l1_simple = [r.reward for r in training_record_multi_l_1_2_seed1]
l2_simple = [r.reward for r in training_record_multi_l_1_2_seed2]
l3_simple = [r.reward for r in training_record_multi_l_1_2_seed3]
l4_simple = [r.reward for r in training_record_multi_l_1_2_seed4]
l5_simple = [r.reward for r in training_record_multi_l_1_2_seed5]
l6_simple = [r.reward for r in training_record_multi_l_1_2_seed6]
l7_simple = [r.reward for r in training_record_multi_l_1_2_seed7]
l8_simple = [r.reward for r in training_record_multi_l_1_2_seed8]
l9_simple = [r.reward for r in training_record_multi_l_1_2_seed9]
l10_simple = [r.reward for r in training_record_multi_l_1_2_seed10]

multiple_lists = [l1_simple, l2_simple, l3_simple, l4_simple, l5_simple, l6_simple, l7_simple, l8_simple, l9_simple, l10_simple]
arrays = [np.array(x) for x in multiple_lists]
_mean_lst = [np.mean(k) for k in zip(*arrays)]
_std_lst = [np.std(g) for g in zip(*arrays)]
_std_shade_pos = [sum(x) for x in zip(_mean_lst, _std_lst)]
_std_shade_neg = [m - n for m,n in zip(_mean_lst, _std_lst)]

plt.plot([r.ep for r in training_record_multi_l_1_2_seed10], mean_lst, color = 'orange')
plt.fill_between( [r.ep for r in training_record_multi_l_1_2_seed10],std_shade_neg, std_shade_pos, color = 'mocassin')

plt.plot([r.ep for r in training_record_multi_l_1_2_seed10], _mean_lst, color = 'royalblue')
plt.fill_between( [r.ep for r in training_record_multi_l_1_2_seed10],_std_shade_neg, _std_shade_pos, color = 'lightsteelblue')
plt.xlabel('Episode')
plt.ylabel('Mean moving averaged reward')
plt.legend(['SMMOT', 'SWPWIL'])
plt.title('Length = 1.2')
plt.savefig("/Users/ilanasebag/Documents/Thesis_code/IL_results/multi_unique_len_1_2_%s.png"%environment)
plt.show()

### Experiment 3 : 5 expert trajectories each of different length

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj1
    seeds = 1
    training_records_diff_len_MMOT_seed1 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj1
    seeds = 2
    training_records_diff_len_MMOT_seed2 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj1
    seeds = 3
    training_records_diff_len_MMOT_seed3 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj1
    seeds = 4
    training_records_diff_len_MMOT_seed4 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj1
    seeds = 5
    training_records_diff_len_MMOT_seed5 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj1
    seeds = 6
    training_records_diff_len_MMOT_seed6 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj1
    seeds = 7
    training_records_diff_len_MMOT_seed7 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj1
    seeds = 8
    training_records_diff_len_MMOT_seed8 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj1
    seeds = 9
    training_records_diff_len_MMOT_seed9 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj1
    seeds = 10
    training_records_diff_len_MMOT_seed10 = main(environment, exp, seeds, simple = False, MMOT = True, wass_PWIL = False, PWIL = False)

l1 = [r.reward for r in training_records_diff_len_MMOT_seed1]
l2 = [r.reward for r in training_records_diff_len_MMOT_seed2]
l3 = [r.reward for r in training_records_diff_len_MMOT_seed3]
l4 = [r.reward for r in training_records_diff_len_MMOT_seed4]
l5 = [r.reward for r in training_records_diff_len_MMOT_seed5]
l6 = [r.reward for r in training_records_diff_len_MMOT_seed6]
l7 = [r.reward for r in training_records_diff_len_MMOT_seed7]
l8 = [r.reward for r in training_records_diff_len_MMOT_seed8]
l9 = [r.reward for r in training_records_diff_len_MMOT_seed9]
l10 = [r.reward for r in training_records_diff_len_MMOT_seed10]

multiple_lists = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10]
arrays = [np.array(x) for x in multiple_lists]
mean_lst = [np.mean(k) for k in zip(*arrays)]
std_lst = [np.std(g) for g in zip(*arrays)]

std_shade_pos = [sum(x) for x in zip(mean_lst, std_lst)]
std_shade_neg = [m - n for m,n in zip(mean_lst, std_lst)]

plt.plot([r.ep for r in training_records_diff_len_MMOT_seed1], mean_lst, color = 'orange')
plt.fill_between( [r.ep for r in training_records_diff_len_MMOT_seed1],std_shade_neg, std_shade_pos, color = 'moccasin')
plt.xlabel('Episode')
plt.title('IL - MMOT - multiple trajectories of different length')
plt.ylabel('Mean moving averaged reward')
plt.legend(['MMOT'])
plt.savefig("/Users/ilanasebag/Documents/Thesis_code/IL_results/multi_diff_trajs_MMOT_%s.png"%environment)

plt.show()

np.save('/Users/ilanasebag/Documents/Thesis_code/IL_results/mean_lst_%s'%environment, mean_lst)
np.save('/Users/ilanasebag/Documents/Thesis_code/IL_results/std_shade_pos_%s'%environment, std_shade_pos)
np.save('/Users/ilanasebag/Documents/Thesis_code/IL_results/std_shade_neg_%s'%environment, std_shade_neg)


if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj1
    seeds = 11
    training_records_diff_len_wass_PWIL_seed1 = main(environment, exp,seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj1
    seeds = 12
    training_records_diff_len_wass_PWIL_seed2 = main(environment, exp,seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj1
    seeds = 13
    training_records_diff_len_wass_PWIL_seed3 = main(environment, exp,seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj1
    seeds = 14
    training_records_diff_len_wass_PWIL_seed4 = main(environment, exp,seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj1
    seeds = 15
    training_records_diff_len_wass_PWIL_seed5 = main(environment, exp,seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj1
    seeds = 16
    training_records_diff_len_wass_PWIL_seed6 = main(environment, exp,seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj1
    seeds = 17
    training_records_diff_len_wass_PWIL_seed7 = main(environment, exp,seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj1
    seeds = 18
    training_records_diff_len_wass_PWIL_seed8 = main(environment, exp,seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj1
    seeds = 19
    training_records_diff_len_wass_PWIL_seed9 = main(environment, exp,seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

if __name__ == '__main__':
    environment = 'Pendulum-v0'
    exp = traj1
    seeds = 20
    training_records_diff_len_wass_PWIL_seed10 = main(environment, exp,seeds, simple = False, MMOT = False, wass_PWIL = True, PWIL = False)

_l1 = [r.reward for r in training_records_diff_len_wass_PWIL_seed1]
_l2 = [r.reward for r in training_records_diff_len_wass_PWIL_seed2]
_l3 = [r.reward for r in training_records_diff_len_wass_PWIL_seed3]
_l4 = [r.reward for r in training_records_diff_len_wass_PWIL_seed4]
_l5 = [r.reward for r in training_records_diff_len_wass_PWIL_seed5]
_l6 = [r.reward for r in training_records_diff_len_wass_PWIL_seed6]
_l7 = [r.reward for r in training_records_diff_len_wass_PWIL_seed7]
_l8 = [r.reward for r in training_records_diff_len_wass_PWIL_seed8]
_l9 = [r.reward for r in training_records_diff_len_wass_PWIL_seed9]
_l10 = [r.reward for r in training_records_diff_len_wass_PWIL_seed10]

_multiple_lists = [_l1, _l2, _l3, _l4, _l5, _l6, _l7, _l8, _l9, _l10]
_arrays = [np.array(x) for x in _multiple_lists]
_mean_lst = [np.mean(k) for k in zip(*_arrays)]
_std_lst = [np.std(g) for g in zip(*_arrays)]

_std_shade_pos = [sum(x) for x in zip(_mean_lst, _std_lst)]
_std_shade_neg = [m - n for m,n in zip(_mean_lst, _std_lst)]

plt.plot([r.ep for r in training_records_diff_len_wass_PWIL_seed1], _mean_lst, color = 'royalblue')
plt.fill_between( [r.ep for r in training_records_diff_len_wass_PWIL_seed1],_std_shade_neg, _std_shade_pos, color = 'lightsteelblue')
plt.xlabel('Episode')
plt.title('IL - WassPWIL - multiple trajectories of different length')
plt.ylabel('Mean moving averaged reward')
plt.legend(['Wass PWIL'])
plt.savefig("/Users/ilanasebag/Documents/Thesis_code/IL_results/multi_diff_trajs_WassPWIL_%s.png"%environment)
plt.show()

np.save('/Users/ilanasebag/Documents/Thesis_code/IL_results/_mean_lst', _mean_lst)
np.save('/Users/ilanasebag/Documents/Thesis_code/IL_results/_std_shade_pos', _std_shade_pos)
np.save('/Users/ilanasebag/Documents/Thesis_code/IL_results/_std_shade_neg', _std_shade_neg)


#MMOT
mean_lst = np.load('/Users/ilanasebag/Documents/Thesis_code/IL_results/mean_lst_%s.npy'%environment)
std_shade_pos = np.load('/Users/ilanasebag/Documents/Thesis_code/IL_results/std_shade_pos_%s.npy'%environment)
std_shade_neg = np.load('/Users/ilanasebag/Documents/Thesis_code/IL_results/std_shade_neg_%s.npy'%environment)

mean_lst = list(mean_lst)
std_shade_pos =list(std_shade_pos)
std_shade_neg = list(std_shade_neg)

#Wass PWIL
_mean_lst = np.load('/Users/ilanasebag/Documents/Thesis_code/IL_results/_mean_lst_%s.npy'%environment)
_std_shade_pos = np.load('/Users/ilanasebag/Documents/Thesis_code/IL_results/_std_shade_pos_%s.npy'%environment)
_std_shade_neg = np.load('/Users/ilanasebag/Documents/Thesis_code/IL_results/_std_shade_neg_%s.npy'%environment)

_mean_lst = list(_mean_lst)
_std_shade_pos =list(_std_shade_pos)
_std_shade_neg = list(_std_shade_neg)

#final summary plot with 10 seeds 

plt.plot([r.ep for r in training_records_diff_len_MMOT_seed1], mean_lst, color = 'orange')
plt.fill_between( [r.ep for r in training_records_diff_len_MMOT_seed1],std_shade_neg, std_shade_pos, color = 'moccasin')

plt.plot([r.ep for r in training_records_diff_len_wass_PWIL_seed1], _mean_lst, color = 'royalblue')
plt.fill_between( [r.ep for r in training_records_diff_len_wass_PWIL_seed1],_std_shade_neg, _std_shade_pos, color = 'lightsteelblue')


plt.xlabel('Episode')
plt.title('Imitation Learning - multiple trajectories of different length')
plt.ylabel('Mean moving averaged reward')
plt.legend(['MMOT', 'Wass PWIL'])
plt.savefig("/Users/ilanasebag/Documents/Thesis_code/IL_results/multi_diff_trajs_comparison_%s.png"%environment)

plt.show()











