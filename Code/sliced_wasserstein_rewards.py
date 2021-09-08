import gym
import numpy as np
import cvxpy as cp
import sys
import pylab
import matplotlib.pyplot as plt
import torch
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
from tqdm.notebook import tqdm




def rand_projections(embedding_dim, num_samples=10):
    projections = [w / np.sqrt((w**2).sum())  # L2 normalization
                   for w in np.random.normal(size=(num_samples, embedding_dim))]
    projections = np.asarray(projections)
    return torch.from_numpy(projections).type(torch.FloatTensor)
   
def rewarder_1d(emp_measures, weights = None, device = 'cpu'):
        # emp_measures: list of measures, each being an array of dim n_support_points x 1
        num_measures = len(emp_measures)
        n_support_points = emp_measures[0].shape[0]
        if weights == None:
            weights = torch.ones(num_measures, dtype=torch.float64)/num_measures
            weights = weights.reshape(-1,1).to(device).float()
           
        # Creates a matrix n_support_points x n_measures --  [mu_1, ... , mu_P].
        emp_measures = torch.cat(emp_measures,1)
        # Sorts the individual measures, still a n_support_points x n_measures matrix, but sorted column-wise.
        emp_measures_sorted = torch.sort(emp_measures,0).values
        # For each point index, compute weighted sum across measures. This is n_support_points x 1.
        sum_x_w = torch.matmul(emp_measures_sorted, weights).reshape(-1,1)
        #Compute matrix of y_n^(p) = x_n^(p)-\sum_j beta_j x_n^(j). This is n_support_points x n_measures matrix
        y = (emp_measures_sorted-sum_x_w)**2
       
         #NP*Wass = sum(rewards)
        rewards = y.clone()
        for i in range(num_measures):
            perm = np.argsort( emp_measures[:,i], 0 )
            inv_perm = np.argsort( perm, 0 )
            rewards[:,i] = y[:,i][inv_perm]
       
        return(rewards)
   
def rewarder_multi(emp_measures, num_projections = 50, weights = None, device = 'cpu'):
    num_measures = len(emp_measures)
    n_support_points = emp_measures[0].shape[0]
    d = emp_measures[0].shape[1]
       
       
    #Matrix of projection directions. It is d x num_projections
    projections = rand_projections(d, num_projections).T
    projections = projections.to(device)
    # Projecting individual measures along num_projections axis. Each measure is now n_support_points x num_projections
    measures_proj=[elt.to(device).matmul(projections) for elt in emp_measures]
    scores=[]
    #For each projection, compute the multimarginal Wasserstein between the projected measures
    for i in range(num_projections):
        #List of measures projected on the i_th projection   TODO: use a torch map to compute stuffs in parallel
        projected_measure_i = [measures_proj[m][:,i].reshape(-1,1) for m in range(num_measures)]
        score_i = rewarder_1d(projected_measure_i, device = device, weights = weights)
        scores.append(score_i)
    scores = torch.stack(scores)
    #Monte-Carlo estimation of the sliced-wasserstein
    rewards = (1/num_projections)*torch.sum(scores, axis = 0)
   
    return rewards
    