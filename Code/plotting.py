import gym
import numpy as np
import cvxpy as cp
import sys
import pylab
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

#Plot Pendulum Trajectory 
##Step by step (200 steps per episode)

def preprocess_states(states):
    theta = np.arctan(np.abs(states[:,1]/states[:,0])) + np.pi/2

    x = np.cos(theta)
    y = np.sin(theta)

    return np.stack((x,y), axis = 1)

def animate(states_nc, legend=False):
    (fig, ax) = plt.subplots(figsize=(8,8))
    fig.patch.set_facecolor((1,1,1))

    gt_circ_nc = matplotlib.patches.Circle((0,0),radius=0.1,zorder=0,color="cornflowerblue")
    
    ax.add_patch(gt_circ_nc)
    
    gt_str_nc = ax.plot((),"-",zorder=1,color="black",linewidth=3)
   
    time = ax.text(-2.22,-2.35, "", ha="left", fontsize=28)

    ax.axis('off')
    ax.set_xlim((-1.2,1.2))
    ax.set_ylim((-0.3,1.2))
    ax.set_aspect('equal', adjustable='box')
    
    if legend:
        handle_nc = matplotlib.lines.Line2D((0,),(0,), marker="o", color="w", markerfacecolor="cornflowerblue", label="Ground Truth", markersize=28)
    
        ax.legend(handles = (handle_nc), framealpha=0.0,ncol=2,loc="center",bbox_to_anchor=(0.5,1.05),handlelength=0.5,fontsize=16)


    def animate(i):
            gt_circ_nc.set_center((states_nc[i,0],states_nc[i,1]))

            gt_str_nc[0].set_data((0,states_nc[i,0]),(0,states_nc[i,1]))

            return (gt_circ_nc, time)

    anim = FuncAnimation(fig, animate, init_func = lambda: animate(0), interval = 20,
                         frames = np.arange(len(states_nc)), blit=True, repeat=True)
    #anim.save('oklm.mp4')
    if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
            return IPython.display.HTML(anim.to_jshtml())
        
    ax.close()
    
    
    