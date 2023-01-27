#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 22:04:45 2022

@author: qiang
"""

import torch
import torch.nn as nn
import utils

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0, std=0.5)
    elif type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        nn.init.uniform_(layer.bias, a=-0.1, b=0.1)

class FilterNet(nn.Module):
    def __init__(self,
                 obs_dim=97,
                 action_dim=9,
                 bias=True,
                 tune=False):

        super(FilterNet, self).__init__()
        self.max_action = utils.RRC_MAX_ACTION
        self.net_obs = nn.Sequential(nn.Linear(obs_dim, 512, bias=bias),
                                     nn.BatchNorm1d(512),
                                     nn.ReLU(),
                                     nn.Linear(512, 512, bias=bias),
                                     nn.BatchNorm1d(512),
                                     nn.ReLU(),
                                     nn.Linear(512, 256, bias=bias),
                                     nn.BatchNorm1d(256),
                                     nn.ReLU(),
                                     nn.Linear(256, 128, bias=bias),
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(),
                                     nn.Linear(128, action_dim*2, bias=bias),
                                     )

        self.net = nn.Sequential(nn.Linear(action_dim*3, 128, bias=bias),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(),
                                 nn.Linear(128, 128, bias=bias),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(),
                                 nn.Linear(128, 2, bias=bias),
                                 nn.Softmax(dim=1),
                                 # nn.Sigmoid()
                                 )

    def forward(self, o, a):
        o = self.net_obs(o)
        x = torch.cat((o, a), 1)
        x = self.net(x)
        return x


class BcNet(nn.Module):
    def __init__(self,
                 obs_dim=97,
                 action_dim=9,
                 bias=True,
                 tune=False):

        super(BcNet, self).__init__()
        self.max_action = 0.397
        self.net = nn.Sequential(nn.Linear(obs_dim, 512, bias=bias),
                                 nn.BatchNorm1d(512),
                                 nn.ReLU(),
                                 nn.Linear(512, 512, bias=bias),
                                 nn.BatchNorm1d(512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256, bias=bias),
                                 nn.BatchNorm1d(256),
                                 nn.ReLU(),
                                 nn.Linear(256, 128, bias=bias),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(),
                                 nn.Linear(128, action_dim, bias=bias),
                                 nn.Tanh(),
                                 )

    def forward(self, x):
        x = self.net(x)
        x = self.max_action * x
        return x
