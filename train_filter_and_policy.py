#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 22:00:42 2022

@author: qiang
"""
import argparse
import utils
import trainer
import buffer2 as buffer
import d3rlpy
import gym
from d3rlpy.algos import PLAS,IQL,TD3PlusBC,BC,AWAC
from d3rlpy.metrics.scorer import evaluate_on_environment
import models
import torch
import numpy as np
import os
import random

def main_filter(args):
    args = utils.device_handler(args)
    args = utils.directory_handler(args)
    args = utils.task_handler(args)
    raw_dataset, anchor_dataset = utils.load_dataset(args)
    buff = buffer.FilterBuffer(args,raw_dataset, anchor_dataset)
    
    classifier = trainer.FilterTrainer(args)
    if args.exist_classifier_path:
        print('LOading the exist classifier')
        classifier.reset()
        classifier.load(args.exist_classifier_path, args.exist_classifier_epoch_num)
        buff.classifier_validate(classifier.model, log_path=f'./save/{args.task}/{classifier.train_logger._experiment_name}/adap_probs_{args.exist_classifier_epoch_num}.jpg')
    else:
        buff.set_anchor()
        classifier.train(buff.torch_loader, args.pre_train_epochs, 'pre', buff.classifier_validate)
        for turn in range(args.buffer_epochs):
            buff.relabel_dataset()
            classifier.train(buff.torch_loader, args.filter_epochs, turn, buff.classifier_validate)
    
    policy_dataset = buff.init_torch_loader_to_train_policy()
    mdpd_dataset = d3rlpy.dataset.MDPDataset(
                                            observations=policy_dataset['observations'],
                                            actions=policy_dataset['actions'],
                                            rewards=policy_dataset['rewards'],
                                            terminals=policy_dataset['timeouts'],
                                        )
    
    print('Training polocy')
    env = gym.make(args.task)
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
    
    env.seed(args.seeds)
    utils.set_seed(args.seeds)
    models.set_seed(args.seeds)
    random.seed(args.seeds)
    os.environ['PYTHONHASHSEED'] = str(args.seeds)
    np.random.seed(args.seeds)
    torch.manual_seed(args.seeds)
    torch.cuda.manual_seed(args.seeds)
    torch.cuda.manual_seed_all(args.seeds)
    
    evaluate_scorer = evaluate_on_environment(env)
    algorithm.fit(
                mdpd_dataset,
                n_steps=1e6,
                scorers={
                    'environment': evaluate_scorer}
                )
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', default="filtered")
    parser.add_argument('--task', default='hopper-medium-expert-v2') # walker2d/halfcheetach
    parser.add_argument('--policy', default='td3bc') #bc,iql,td3bc,awac
    parser.add_argument('--model-num', default=5)
    parser.add_argument('--ensemble-method', default='avg')
    parser.add_argument('--buffer-epochs', default=2)
    parser.add_argument('--pre-train-epochs', default=20)
    parser.add_argument('--filter-epochs', default=20)
    parser.add_argument('--anchor-rate', default=0.001)
    parser.add_argument('--raw-dataset-path', default=None)
    parser.add_argument('--anchor-dataset-path', default=None)
    
    parser.add_argument('--th-bins', default=100)
    parser.add_argument('--th-high-bound', default=0.96)
    parser.add_argument('--th-fit-pow', default=10)
    
    parser.add_argument('--seeds', default=123)
    parser.add_argument('--use-gpu', default=1)
    parser.add_argument('--raw-batch-size', default=1024)
    parser.add_argument('--filter-batch-size', default=1024)
    parser.add_argument('--filter-learning-rate', default=0.001)
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--exist-classifier-path', default=None)
    parser.add_argument('--exist-classifier-epoch-num', default=2)
    args = parser.parse_args()
    main_filter(args)