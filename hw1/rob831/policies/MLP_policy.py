import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from rob831.infrastructure import pytorch_util as ptu
from rob831.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    #checking if batched
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
          #adds dimension
            observation = obs[None]
        observation = ptu.from_numpy(observation)

      # TODO return the action that the policy prescribes
        return  ptu.to_numpy(self.forward(observation)) # from tensor to numpy

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        # zero gradients
        self.optimizer.zero_grad()
  
        pred = self.forward(observations)
        # self.get_action(ptu.to_numpy(observations))
        ground_t = actions
        out_put = self.loss(pred, ground_t)

        #backprogradiation
        out_put.backward()

        #optimize loss
        self.optimizer.step()

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> Any: # return tensor
      # print(self.logits_na(observation))
      # print('here', observation)
      # print('there', self.mean_net(observation))
      #if discrete --> logits
      if self.discrete:
        # print(self.logits_na(observation))
        return self.logits_na(observation)
      else:
        # print('here', observation.device)
        # observation = observation.to(device)
        # print('there', self.mean_net.device)
        return self.mean_net(observation)
      # if continuous, mean

#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.loss = nn.MSELoss()

    def update(
            self, observations, actions,
            adv_n=None, acs_labels_na=None, qvals=None
    ):
    
        # TODO: update the policy and return the loss
        # zero gradients
        self.optimizer.zero_grad()

        pred = self.forward(observations)
        # self.get_action(observations)
        ground_t = actions
        out_put = self.loss(pred, ground_t)

        #backprogradiation
        out_put.backward()

        #optimize loss
        self.optimizer.step()
        
        # outloss = update(observations, actions)
         
        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(out_put),
        }

#####################################################
####### DRAFT
#####################################################
        #forward
        # model = self.update(self.observations, self.actions)

        # output = self.forward(model)

        # model = ptu.build_mlp(self.observations.shape[0],
        # self.actions.shape[0],
        # self.n_layers,
        # self.size,
        # )

        #loss:
        # loss = self.loss 
        # loss = nn.MSELoss()

      # output = ptu.build_mlp(self.observation)
      # model = nn.Sequential(output)
      # observation = F.relu(self.layer1(observation))
      # observation = F.relu(self.layer2(observation))
      # output = F.identity(observation)
      # action = get_action
      # model
      # return output
      # model = ptu.build_mlp(self.input_size, 
      #   self.output_size, 
      #   self.n_layers, 
      #   self.size,
      #   )
      #   state = model.layer
      # state = nn.Linear(self.input_size, self.size)
      # state
      # observation = F.tanh()
      # observation = F.identity(nn.Linear(self.size, self.output_size))
      # observation = 
      # x = self.flatten(observation)


      # logits = ptu.build_mlp(x.shape[0], self.output_size, self.n_layers, self.size)
      # return logits
      # model = ptu.build_mlp(self.input_size, 
      #   self.output_size, 
      #   self.n_layers, 
      #   self.size,
      # )
      # return 

        # raise NotImplementedError


        
      #forward --> forward pass of NN
      # 
      # return self.forward(self.observations)


      #goal: train policy for action given observation
        # input_size = self.observations.shape[0]
        # output_size = self.actions.shape[0]
        # model = ptu.build_mlp(input_size,
        #  output_size, 
        #   self.n_layers, self.size,
        # )
        # output = nn.Sequential(model)
        # return output

      
        # self.Env.step(observations)
        # a = self.get_action(observations) 
        # returns actions
        # raise NotImplementedError


                
        # model = ptu.build_mlp(self.input_size, 
        # self.output_size, 
        # self.n_layers, 
        # self.size,
        # )
    


        # raise NotImplementedError


