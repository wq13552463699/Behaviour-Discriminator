#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 11:22:44 2022

@author: qiang
"""
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import os
import numpy as np
import torch
from typing_extensions import Protocol
from typing import Any, Dict, Iterator, List, Optional
import structlog
from tensorboardX import SummaryWriter
from datetime import datetime
import time
import json
from contextlib import contextmanager
from d3rlpy.dataset import Episode, MDPDataset, Transition
from typing import (cast, List)
from d3rlpy.iterators import RoundIterator
import math
from copy import deepcopy
import gc
import gym
import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
import gym, d4rl
import numpy as np

RRC_LIFT_TASK_OBS_DIM = 139
RRC_PUSH_TASK_OBS_DIM = 97
RRC_ACTION_DIM = 9
RRC_MAX_ACTION = 0.397

def device_handler(args):
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device('cuda')
        print('Using CUDA')
    else:
        device = torch.device('cpu')
    args.device = device
    return args

def directory_handler(args):
    if not args.save_path:
        proj_root_path = os.path.split(os.path.realpath(__file__))[0]
        args.save_path = f'{proj_root_path}/save'
    if os.path.split(args.save_path)[-1] != args.task:
        args.save_path = f'{args.save_path}/{args.task}'
    return args

def load_dataset(args):
    if args.raw_dataset_path:
        raw_dataset = np.load(args.raw_dataset_path, allow_pickle=True).item()
    else:
        raw_dataset = gym.make(args.task).get_dataset()
        
    if args.anchor_dataset_path:
        anchor_dataset = np.load(args.anchor_dataset_path, allow_pickle=True).item()
    else:
        anchor_dataset = None
    return raw_dataset, anchor_dataset

def adap_probs(probs, bins=100, fit_pow=8, prob_th=0.96, plot=True, save_plot_path = None):
    def get_minima(values: np.ndarray):
        min_index = sg.argrelmin(values)[0]
        return min_index, values[min_index]

    def ydata_gen(probs):
        return np.histogram(probs,bins=bins)[0]
    
    unit = (probs.max() - probs.min()) / bins
    xdata = []
    for i in range(bins):
        xdata.append(probs.min() + (i * unit))
    xdata = np.array(xdata)
    ydata = np.array(ydata_gen(probs))
    print(probs.max())
    print(probs.min())
    
    coeffi = np.polyfit(xdata, ydata, fit_pow)
    pln = np.poly1d(coeffi)
    y_pred=pln(xdata)

    idxs, values = get_minima(y_pred)

    if idxs.shape[0] > 1:
        if xdata[idxs[-1]] >= prob_th:
            adap_p = xdata[idxs[-2]]
            adap_v = values[-2]
        else:
            adap_p = xdata[idxs[-1]]
            adap_v = values[-1]
    else:
        if xdata[idxs[-1]] >= prob_th:
            adap_p = prob_th
            adap_v = None
        else:
            adap_p = xdata[idxs[-1]]
            adap_v = values[-1]

    plt.figure()
    
    plt.plot(xdata, ydata, '*',color='gold',label='original values')
    plt.plot(xdata, y_pred, color='r',label='polyfit values')
    plt.scatter(adap_p, adap_v, s=40,marker='D',color='b', label='adapted point')
    plt.legend(loc=0)
    if plot:
        plt.show()
    if save_plot_path:
        plt.savefig(save_plot_path)
    print(f'The adaptive probability is {adap_p}')
    return adap_p

def task_handler(args):
    if args.task == 'rrc_lift_mix':
        args.task_type = 'lift'
        args.diff = 'mixed'
        args.obs_dim = 139
        args.action_dim = 9
        args.half = 1197
        args.use_terminal = False
    elif args.task == 'rrc_push_mix':
        args.task_type = 'push'
        args.diff = 'mixed'
        args.obs_dim = 97
        args.action_dim = 9
        args.half = 1920
        args.use_terminal = False
    elif args.task == 'ant':
        args.task_type = 'ant'
        args.diff = 'mixed'
        args.obs_dim = 27
        args.action_dim = 8
        args.half = None
        args.use_terminal = False
    elif args.task == 'humanoid':
        args.task_type = 'humanoid'
        args.diff = 'mixed'
        args.obs_dim = 376
        args.action_dim = 17
        args.half = None
        args.use_terminal = False
    elif args.task == 'halfcheetah-medium-replay-v2' or args.task == 'halfcheetah-medium-expert-v2':
        args.task_type = 'halfcheetah'
        args.diff = 'mixed'
        args.obs_dim = 17
        args.action_dim = 6
        args.half = None
        args.use_terminal = False
    elif args.task == 'hopper-medium-expert-v2':
        args.task_type = 'hopper'
        args.diff = 'mixed'
        args.obs_dim = 11
        args.action_dim = 3
        args.half = None
        args.use_terminal = True
    elif args.task == 'walker2d-medium-expert-v2':
        args.task_type = 'walker2d'
        args.diff = 'mixed'
        args.obs_dim = 17
        args.action_dim = 6
        args.half = None
        args.use_terminal = True
    else:
        raise RuntimeError('Invalid input task')
    return args


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torch_loader(raw_dataset,
                 train_batch_size,
                 shuffle,
                 ):
    dataset = TorchDatasetHandler(
        raw_dataset, raw_dataset, transform=None)

    train_loader = DataLoader(
        dataset, batch_size=train_batch_size, shuffle=shuffle)
    return train_loader

class TorchDatasetHandler(Dataset):
    def __init__(self, filtered_dataset, raw_dataset, transform=None):
        self.transform = transform
        self.raw_dataset = raw_dataset
        self.raw_len = self.raw_dataset['observations'].shape[0]

    def __len__(self):
        return self.raw_dataset['actions'].shape[0]

    def __getitem__(self, index):
        _observation = self.raw_dataset['observations'][index]
        _action = self.raw_dataset['actions'][index]
        _label = self.raw_dataset['labels'][index]

        if self.transform:
            _observation = self.transform(_observation)
            _action = self.transform(_action)
            _label = self.transform(_label)

        data = [_observation, _action, _label]
        return data
    
class conf_matrix():
    def __init__(self,
                 ):
        self.title = ['TURN','FILTERED NUM']
        self.data = []
        
    def update(self,log):
        self.data.append(log)
    
    def save(self,path):
        dataframe = pd.DataFrame(data=self.data,columns=self.title)
        dataframe.to_csv(path,index=False,sep=',')
        
    def clear(self):
        self.data=[]
        
class _SaveProtocol(Protocol):
    def save_model(self, fname: str) -> None:
        ...

# default json encoder for numpy objects


def default_json_encoder(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise ValueError(f"invalid object type: {type(obj)}")


LOG: structlog.BoundLogger = structlog.get_logger(__name__)


class TrainLogger:

    _experiment_name: str
    _logdir: str
    _save_metrics: bool
    _verbose: bool
    _metrics_buffer: Dict[str, List[float]]
    _params: Optional[Dict[str, float]]
    _writer: Optional[SummaryWriter]

    def __init__(
        self,
        experiment_name: str,
        tensorboard_dir: Optional[str] = None,
        save_metrics: bool = True,
        root_dir: str = "logs",
        verbose: bool = True,
        with_timestamp: bool = True,
    ):
        self._save_metrics = save_metrics
        self._verbose = verbose

        # add timestamp to prevent unintentional overwrites
        while True:
            if with_timestamp:
                date = datetime.now().strftime("%Y%m%d%H%M%S")
                self._experiment_name = experiment_name + "_" + date
            else:
                self._experiment_name = experiment_name

            if self._save_metrics:
                self._logdir = os.path.join(root_dir, self._experiment_name)
                if not os.path.exists(self._logdir):
                    os.makedirs(self._logdir)
                    LOG.info(f"Directory is created at {self._logdir}")
                    break
                if with_timestamp:
                    time.sleep(1.0)
                if os.path.exists(self._logdir):
                    LOG.warning(
                        f"You are saving another logger into {self._logdir}, this may cause unintentional overite")
                    break
            else:
                break

        self._metrics_buffer = {}

        if tensorboard_dir:
            tfboard_path = self._logdir
            self._writer = SummaryWriter(logdir=tfboard_path)
        else:
            self._writer = None

        self._params = None

    def add_params(self, params: Dict[str, Any]) -> None:
        assert self._params is None, "add_params can be called only once."

        if self._save_metrics:
            # save dictionary as json file
            params_path = os.path.join(self._logdir, "params.json")
            with open(params_path, "w") as f:
                json_str = json.dumps(
                    params, default=default_json_encoder, indent=2
                )
                f.write(json_str)

            if self._verbose:
                LOG.info(
                    f"Parameters are saved to {params_path}", params=params
                )
        elif self._verbose:
            LOG.info("Parameters", params=params)

        # remove non-scaler values for HParams
        self._params = {k: v for k, v in params.items() if np.isscalar(v)}

    def add_metric(self, name: str, value: float) -> None:
        if name not in self._metrics_buffer:
            self._metrics_buffer[name] = []
        self._metrics_buffer[name].append(value)

    def commit(self, epoch: int, step: int) -> Dict[str, float]:
        metrics = {}
        for name, buffer in self._metrics_buffer.items():

            metric = sum(buffer) / len(buffer)

            if self._save_metrics:
                path = os.path.join(self._logdir, f"{name}.csv")
                with open(path, "a") as f:
                    print(f"{epoch},{step},{metric}", file=f)

                if self._writer:
                    self._writer.add_scalar(f"metrics/{name}", metric, epoch)

            metrics[name] = metric

        if self._verbose:
            LOG.info(
                f"{self._experiment_name}: epoch={epoch} step={step}",
                epoch=epoch,
                step=step,
                metrics=metrics,
            )

        if self._params and self._writer:
            self._writer.add_hparams(
                self._params,
                metrics,
                name=self._experiment_name,
                global_step=epoch,
            )

        # initialize metrics buffer
        self._metrics_buffer = {}
        return metrics

    def save_model(self, epoch: int, algo: _SaveProtocol) -> None:
        if self._save_metrics:
            # save entire model
            model_path = os.path.join(self._logdir, f"model_{epoch}.pt")
            algo.save_model(model_path)
            LOG.info(f"Model parameters are saved to {model_path}")

    def close(self) -> None:
        if self._writer:
            self._writer.close()

    @contextmanager
    def measure_time(self, name: str) -> Iterator[None]:
        name = "time_" + name
        start = time.time()
        try:
            yield
        finally:
            self.add_metric(name, time.time() - start)

    @property
    def logdir(self) -> str:
        return self._logdir

    @property
    def experiment_name(self) -> str:
        return self._experiment_name