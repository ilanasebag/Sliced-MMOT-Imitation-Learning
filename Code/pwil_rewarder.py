# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.




#pip install POT --> python optimal transport library 
import copy
import random
# import dm_env
import numpy as np
import ot
from sklearn import preprocessing
import sklearn 

import torch 


class PWILRewarder(object):
  """Rewarder class to compute PWIL rewards."""

#from here, expert demonstrations has to be taken as input
# they must alredy be concatenated and sampled 
# they must be within a np array and not a tensor !!
# they must be called vectorized_demonstrations
 

  def __init__(self,
               demonstrations,
               obs_act,
               env,
               num_demonstrations=1,
               time_horizon=1000.,
               alpha=5.,
               beta=5.,
               observation_only=False):

    self.num_demonstrations = num_demonstrations
    self.time_horizon = time_horizon

    # Observations and actions are flat.
    dim_act = env.action_space.shape[0]
    dim_obs = env.observation_space.shape[0]
    self.reward_sigma = beta * time_horizon / np.sqrt(dim_act + dim_obs)
    self.reward_scale = alpha

    self.observation_only = observation_only

    self.vectorized_demonstrations = demonstrations
    self.scaler = self.get_scaler()
    


    self.obs_act = obs_act 
    self.expert_atoms = copy.deepcopy(
        self.scaler.transform(self.vectorized_demonstrations)
    )
    self.expert_weights = np.ones(len(self.expert_atoms)) / (len(self.expert_atoms))
    

     
  def get_scaler(self):
    """Defines a scaler to derive the standardized Euclidean distance."""
    scaler = preprocessing.StandardScaler()
    scaler.fit(self.vectorized_demonstrations)
    return scaler

  def reset(self):
    """Makes all expert transitions available and initialize weights."""
    self.expert_atoms = copy.deepcopy(
        self.scaler.transform(self.vectorized_demonstrations)
    )
    num_expert_atoms = len(self.expert_atoms)
    self.expert_weights = np.ones(num_expert_atoms) / (num_expert_atoms)
    
    
  def compute_reward(self):
    """Computes reward as presented in Algorithm 1."""
    # Scale observation and action.
    
    #if self.observation_only:
      #agent_atom = obs_act['observation']
    #else:
      #agent_atom = np.concatenate([obs_act['observation'], obs_act['action']])
    
    for i in range(len(self.obs_act)): 
        lst = []
        agent_atom = np.expand_dims(self.obs_act[i], axis=0)  # add dim for scaler
        #agent_atom = self.scaler.transform(agent_atom)[0]
    
        cost = 0.
        # As we match the expert's weights with the agent's weights, we might
        # raise an error due to float precision, we substract a small epsilon from
        # the agent's weights to prevent that.
        weight = 1. / self.time_horizon - 1e-6
        norms = np.linalg.norm(self.expert_atoms - agent_atom, axis=1)
        while weight > 0:
          # Get closest expert state action to agent's state action.
          argmin = norms.argmin()
          expert_weight = self.expert_weights[argmin]
    
          # Update cost and weights.
          if weight >= expert_weight:
            weight -= expert_weight
            cost += expert_weight * norms[argmin]
            self.expert_weights = np.delete(self.expert_weights, argmin, 0)
            self.expert_atoms = np.delete(self.expert_atoms, argmin, 0)
            norms = np.delete(norms, argmin, 0)
          else:
            cost += weight * norms[argmin]
            self.expert_weights[argmin] -= weight
            weight = 0
    
        reward = self.reward_scale * np.exp(-self.reward_sigma * cost)
        reward = reward.astype('float32')
        lst.append(reward)

    return lst



    def compute_w2_dist_to_expert(self, trajectory):
         """Computes Wasserstein 2 distance to expert demonstrations."""
         self.reset()
         if self.observation_only:
           trajectory = [t['observation'] for t in trajectory]
         else:
           trajectory = [np.concatenate([t['observation'], t['action']])
                         for t in trajectory]
    
         trajectory = self.scaler.transform(trajectory)
         trajectory_weights = 1./len(trajectory) * np.ones(len(trajectory))
         cost_matrix = ot.dist(trajectory, self.expert_atoms, metric='euclidean')
         w2_dist = ot.emd2(trajectory_weights, self.expert_weights, cost_matrix)
         
         return w2_dist
  
    
  
    #based on the sliced wass rewarder file 
    
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
   
def rp(emp_measures, weights = None, device = 'cpu'):
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
    

    