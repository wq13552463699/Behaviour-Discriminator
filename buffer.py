#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 00:32:51 2022

@author: qiang
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import gc
import utils
import random
import math
from copy import copy

def get_anchor(dataset, percent=0.02):
    avg_episodic_rewards = []
    temp_rewards = []
    count = 0
    for idx, timeout in enumerate(dataset['timeouts']):
        temp_rewards.append(dataset['rewards'][idx])
        if timeout:
            avg = np.array(copy(temp_rewards)).mean()
            avg_episodic_rewards.append(avg)
            temp_rewards = []
            count += 1
    anchor_num = int(percent * count)
    avg_episodic_rewards.sort()
    reward_anchor = avg_episodic_rewards[-anchor_num]
    
    anchor_obs = []
    anchor_actions = []
    temp_rewards = []
    temp_actions = []
    temp_obs = []
    num_selected = 0
    for idx, timeout in enumerate(dataset['timeouts']):
        temp_rewards.append(dataset['rewards'][idx])
        temp_obs.append(dataset['observations'][idx].tolist())
        temp_actions.append(dataset['actions'][idx].tolist())
        if timeout:
            avg = np.array(copy(temp_rewards)).mean()
            if avg >= reward_anchor:
                num_selected += 1
                anchor_obs += temp_obs
                anchor_actions += temp_actions
            temp_obs = []
            temp_actions = []
            temp_rewards = []
    
    print(f'{num_selected}/{count} anchors are selected')
    temp_dataset = {}
    temp_dataset['observations'] = np.array(anchor_obs)
    temp_dataset['actions'] = np.array(anchor_actions)
    return temp_dataset

class FilterBuffer():
    def __init__(self,
                 args,
                 raw_dataset,
                 anchor_dataset=None,
                 ):
        self.raw_dataset = raw_dataset
        
        if args.use_terminal:
            print('Converting terminals to timeouts')
            new_timeouts = []
            for idx, timeout in enumerate(self.raw_dataset['timeouts']):
                new_timeouts.append(timeout or self.raw_dataset['terminals'][idx])
            self.raw_dataset['timeouts'] = new_timeouts
        
        if not anchor_dataset:
            print('No anchor dataset is given, get the anchor from the raw dataset')
            anchor_dataset = get_anchor(self.raw_dataset, args.anchor_rate)
        
        self.anchor_dataset = anchor_dataset
        self.anchor_len = self.anchor_dataset['observations'].shape[0]
        self.raw_len = self.raw_dataset['observations'].shape[0]
        self.args = args
        self.torch_loader = None
        self.th_conf = None
        self.args.max_action = self.raw_dataset['actions'].max()
        print(f"max_action: {self.args.max_action}")
        
    def set_anchor(self):
        self.torch_loader = []
        for i in range(self.args.model_num):
            dataset = TorchDatasetHandler(
                self.sub_achor(), self.raw_dataset, self.args, transform=None)
            self.torch_loader.append(DataLoader(
                dataset, batch_size=self.args.filter_batch_size, shuffle=True))
            
    def sub_achor(self):
        rand_idx = random.sample(range(0,self.anchor_len),int(self.anchor_len*0.95))
        sub_obs = self.anchor_dataset['observations'][rand_idx]
        sub_act = self.anchor_dataset['actions'][rand_idx]
        subset = {}
        subset['observations'] = sub_obs
        subset['actions'] = sub_act
        return subset
    
    def reset(self):
        del self.buffers, self.torch_loader
        gc.collect()
    
    def relabel_dataset(self):
        num_in_each = math.floor(self.relabel_num / self.args.model_num)
        print(f'{self.relabel_num}/{self.raw_len} samples in the new dataset')
        self.buffers = []
        full_idx = 0
        sub_len = 0
        subset = {'observations': [],'actions':[]}
        for idx, lab in enumerate(self.relabel):
            if lab:
                subset['observations'].append(self.raw_dataset['observations'][idx])
                subset['actions'].append(self.raw_dataset['actions'][idx])
                sub_len += 1
                if sub_len >= num_in_each:
                    subset['observations'] = np.array(subset['observations'])
                    subset['actions'] = np.array(subset['actions'])
                    self.buffers.append(subset)
                    full_idx += 1
                    sub_len = 0
                    subset = {'observations': [],'actions':[]}
                if full_idx >= self.args.model_num:
                    break
        self._init_torch_loaders()
        
    def init_torch_loader_to_train_policy(self):
        train_set = {'observations': [],'actions':[], 'timeouts':[], 'rewards':[]}
        for idx, lab in enumerate(self.relabel):
            if lab:
                train_set['observations'].append(self.raw_dataset['observations'][idx])
                train_set['actions'].append(self.raw_dataset['actions'][idx])
                train_set['timeouts'].append(self.raw_dataset['timeouts'][idx])
                train_set['rewards'].append(self.raw_dataset['rewards'][idx])
        train_set['observations'] = np.array(train_set['observations'])
        train_set['actions'] = np.array(train_set['actions'])
        train_set['timeouts'] = np.array(train_set['timeouts'])
        train_set['rewards'] = np.array(train_set['rewards'])
        print(f'{self.relabel_num}/{self.raw_len} samples in the new dataset')
        return train_set
    
    def _init_torch_loaders(self):
        self.torch_loader = []
        
        for i in range(self.args.model_num):
            dataset = TorchDatasetHandler(
                self.buffers[i], self.raw_dataset, self.args, transform=None)
            self.torch_loader.append(DataLoader(
                dataset, batch_size=self.args.filter_batch_size, shuffle=True))
        
    def classifier_validate(self, classifiers, log_path=None):
        with torch.no_grad():
            temp_obs = []
            temp_actions = []
            probs = []
            count = 0
            self.relabel = []
            self.relabel_num = 0
            self.relabelled_idx = []

            for idx, timeout in enumerate(self.raw_dataset['timeouts']):
                
                temp_obs.append(self.raw_dataset['observations'][idx].tolist())
                temp_actions.append(self.raw_dataset['actions'][idx].tolist())
    
                if timeout:
                    if torch.cuda.is_available() and self.args.use_gpu:
                        temp_obs_tensor = torch.tensor(temp_obs.copy()).cuda(
                            non_blocking=True).to(torch.float32)
                        temp_actions_tensor = torch.tensor(temp_actions.copy()).cuda(
                            non_blocking=True).to(torch.float32)
                    else:
                        temp_obs_tensor = torch.tensor(temp_obs.copy()).to(
                            torch.float32).to(torch.device('cpu'))
                        temp_actions_tensor = torch.tensor(temp_actions.copy()).to(
                            torch.float32).to(torch.device('cpu'))
                    
                    temp_prob = 0
                    temp_count = 0
                    for classifier in classifiers:
                        classifier.eval()
                        prob = classifier(temp_obs_tensor, temp_actions_tensor).cpu(
                        ).detach().numpy()[..., 0].mean()
                        temp_prob += prob
                        temp_count += 1
                    probs.append(temp_prob/temp_count)
                    temp_obs = []
                    temp_actions = []
            
            #TODO put the arugments into the args
            adap_th = utils.adap_probs(np.array(probs), 
                                       bins=self.args.th_bins, 
                                       fit_pow=self.args.th_fit_pow, 
                                       prob_th=self.args.th_high_bound, 
                                       plot=False, 
                                       save_plot_path = log_path)
                    
            self.th_conf = adap_th
                    
            for idx, timeout in enumerate(self.raw_dataset['timeouts']):
                
                temp_obs.append(self.raw_dataset['observations'][idx].tolist())
                temp_actions.append(self.raw_dataset['actions'][idx].tolist())
    
                if timeout:
                    if torch.cuda.is_available() and self.args.use_gpu:
                        temp_obs_tensor = torch.tensor(temp_obs.copy()).cuda(
                            non_blocking=True).to(torch.float32)
                        temp_actions_tensor = torch.tensor(temp_actions.copy()).cuda(
                            non_blocking=True).to(torch.float32)
                    else:
                        temp_obs_tensor = torch.tensor(temp_obs.copy()).to(
                            torch.float32).to(torch.device('cpu'))
                        temp_actions_tensor = torch.tensor(temp_actions.copy()).to(
                            torch.float32).to(torch.device('cpu'))
                        
                    if self.args.ensemble_method == 'vote':
                        votes = 0
                        for classifier in classifiers:
                            classifier.eval()
                            prob = classifier(temp_obs_tensor, temp_actions_tensor).cpu(
                            ).detach().numpy()[..., 0].mean()
                            votes += float(prob >= self.th_conf)
                        if votes >= len(classifiers) / 2:
                            cond = True
                        else:
                            cond = False
                    
                    if self.args.ensemble_method == 'avg':
                        temp_prob = 0
                        cnt = 0
                        for classifier in classifiers:
                            classifier.eval()
                            prob = classifier(temp_obs_tensor, temp_actions_tensor).cpu(
                            ).detach().numpy()[..., 0].mean()
                            temp_prob += prob
                            cnt += 1
                        avg_prob = temp_prob / cnt
                        if avg_prob >= self.th_conf:
                            cond = True
                        else:
                            cond = False
                        
                    if cond:
                        self.relabel += [1.0] * len(temp_obs)
                        self.relabel_num += len(temp_obs)
                        self.relabelled_idx.append(count)
                    else:
                        self.relabel += [0.0] * len(temp_obs)
                        
                    temp_obs = []
                    temp_actions = []
                    count += 1
        
        print('Relabelling')
        self.relabel = np.array(self.relabel)
        assert self.relabel.shape[0] == self.raw_dataset['observations'].shape[0], "Error"
        
        return len(self.relabelled_idx), count, probs
    
class TorchPolicyDatasetHandler(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return self.dataset['actions'].shape[0]

    def __getitem__(self, index):
        observation = self.dataset['observations'][index]
        action = self.dataset['actions'][index]
        if self.transform:
            observation = self.transform(observation)
            action = self.transform(action)
        data = [observation, action]
        return data

class TorchDatasetHandler(Dataset):
    def __init__(self, filtered_dataset, raw_dataset, args=None, transform=None):
        self.args = args
        self.dataset = filtered_dataset
        self.transform = transform
        self.raw_dataset = raw_dataset
        self.raw_len = self.raw_dataset['observations'].shape[0]

    def __len__(self):
        return self.dataset['actions'].shape[0]

    def __getitem__(self, index):
        pos_observation = self.dataset['observations'][index]
        pos_action = self.dataset['actions'][index]

        if np.random.uniform(0, 1) <= 0.5:
            action = pos_action
            observation = pos_observation
            label = np.array([1, 0])
        else:
            label = np.array([0, 1])
            observation = pos_observation
            if np.random.uniform(0, 1) < 0.5:  # Random actions
                random_action = np.random.uniform(-self.args.max_action, self.args.max_action, size=self.args.action_dim)
                action = random_action
            else:
                while True:
                    random_num = np.random.randint(0, self.raw_len)
                    # print(pos_action)
                    # print(self.raw_dataset["actions"][random_num])
                    # print(pos_action == self.raw_dataset["actions"][random_num])
                    if not (pos_action == self.raw_dataset["actions"][random_num]).all():
                        action = self.raw_dataset["actions"][random_num]
                        break
# TODOs: add negative sample generator from another script in this one
# TODOs: the current negative sample generator is sufficient to train D4RL tasks. 

        if self.transform:
            observation = self.transform(observation)
            action = self.transform(action)
            label = self.transform(label)

        data = [observation, action, label]
        return data