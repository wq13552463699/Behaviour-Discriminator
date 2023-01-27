#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 13:47:45 2022

@author: qiang
"""
import argparse
import utils
import d3rlpy
import gym
from d3rlpy.algos import PLAS,IQL,TD3PlusBC,BC,AWAC
from d3rlpy.metrics.scorer import evaluate_on_environment
import models
import torch
import numpy as np
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument('--exp-name', default="non-filtered")
parser.add_argument('--task', default='hopper-medium-expert-v2') # walker2d/halfcheetach/hopper
parser.add_argument('--policy', default='plas')
parser.add_argument('--seeds', default=114)
parser.add_argument('--raw-dataset-path', default=None)
parser.add_argument('--anchor-dataset-path', default=None)
args = parser.parse_args()

raw_dataset, _ = utils.load_dataset(args)

mdpd_dataset = d3rlpy.dataset.MDPDataset(
                                        observations=raw_dataset['observations'],
                                        actions=raw_dataset['actions'],
                                        rewards=raw_dataset['rewards'],
                                        terminals=raw_dataset['timeouts'],
                                    )
env = gym.make(args.task)
env.seed(args.seeds)
def set_seed(seed):
    utils.set_seed(seed)
    models.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if args.policy == 'bc':
    algorithm = BC(use_gpu=True)
    algorithm.build_with_dataset(mdpd_dataset)

elif args.policy == 'plas':
    algorithm = PLAS(use_gpu=True)
    algorithm.build_with_dataset(mdpd_dataset)
    
elif args.policy == 'td3bc':
    algorithm = TD3PlusBC(use_gpu=True)
    algorithm.build_with_dataset(mdpd_dataset)

elif args.policy == 'iql':
    algorithm = IQL(use_gpu=True)
    algorithm.build_with_dataset(mdpd_dataset)
    
elif args.policy == 'awac':
    algorithm = AWAC(use_gpu=True)
    algorithm.build_with_dataset(mdpd_dataset)
else:
    raise RuntimeError('Error')

evaluate_scorer = evaluate_on_environment(env)
algorithm.fit(mdpd_dataset,
            n_steps=1e6,
            scorers={
                'environment': evaluate_scorer}
            )