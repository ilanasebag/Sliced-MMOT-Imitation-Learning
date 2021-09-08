
from itertools import count
from PIL import Image
import torch
import torch.optim 
import torchvision.transforms as transforms
from collections import namedtuple, deque
from torch import nn
from gym import make
import torch.optim as optim
from tqdm.notebook import tqdm
import numpy as np
import gym


def concatenate_and_sample(exp):


    exp1 = exp[0]
    exp2 = exp[1]
    exp3 = exp[2]
    exp4 = exp[3]
    exp5 = exp[4]
    
    states = np.zeros(shape=(200,3))
    
    for j in range(200):
        for i in range(3):
            states[j,i] = np.random.choice([exp1[j,i], exp2[j,i], exp3[j,i], exp4[j,i], exp5[j,i] ], replace = True)
            
    return(states)




def concatenate_and_sample_cartpole(exp):

    exp1 = exp[0]
    exp2 = exp[1]
    exp3 = exp[2]
    exp4 = exp[3]
    exp5 = exp[4]
    
    states = np.zeros(shape=(200,4))
    
    for j in range(200):
        for i in range(4):
            states[j,i] = np.random.choice([exp1[j,i], exp2[j,i], exp3[j,i], exp4[j,i], exp5[j,i] ], replace = False)
            
    return(states)
    

def get_agent_reward(reward_sw):
    agent_reward = reward_sw[:,:1]
    return agent_reward