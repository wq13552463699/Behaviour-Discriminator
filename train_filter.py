#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 22:00:42 2022

@author: qiang
"""
import argparse
import utils
import trainer
import buffer

def main_filter(args):
    args = utils.device_handler(args)
    args = utils.directory_handler(args)
    args = utils.task_handler(args)

    raw_dataset, anchor_dataset = utils.load_dataset(args)

    classifier = trainer.FilterTrainer(args)
    if args.exist_classifier_path:
        classifier.load(args.exist_classifier_path, args.exist_classifier_epoch_num)
    buff = buffer.FilterBuffer(args,raw_dataset, anchor_dataset)

    buff.set_anchor()
    classifier.train(buff.torch_loader, args.pre_train_epochs, -1, buff.classifier_validate)
    
    for turn in range(args.buffer_epochs):
        buff.relabel_dataset()
        classifier.train(buff.torch_loader, args.filter_epochs, turn, buff.classifier_validate)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', default="halfcheetah")
    parser.add_argument('--task', default='halfcheetah-medium-expert-v2') # walker2d/halfcheetach/hopper
    parser.add_argument('--model-num', default=5)
    parser.add_argument('--ensemble-method', default='avg')
    parser.add_argument('--buffer-epochs', default=30)
    parser.add_argument('--pre-train-epochs', default=20)
    parser.add_argument('--filter-epochs', default=20)
    parser.add_argument('--anchor-rate', default=0.01)
    parser.add_argument('--raw-dataset-path', default=None)
    parser.add_argument('--anchor-dataset-path', default=None)
    
    parser.add_argument('--th-bins', default=100)
    parser.add_argument('--th-high-bound', default=0.96)
    parser.add_argument('--th-fit-pow', default=10)
    
    parser.add_argument('--seeds', default=0)
    parser.add_argument('--use-gpu', default=1)
    parser.add_argument('--raw-batch-size', default=1024)
    parser.add_argument('--filter-batch-size', default=1024)
    parser.add_argument('--filter-learning-rate', default=0.001)
    parser.add_argument('--save-path', default=None)
    parser.add_argument('--exist-classifier-path', default=None)
    parser.add_argument('--exist-classifier-epoch-num', default=None)
    args = parser.parse_args()
    main_filter(args)
