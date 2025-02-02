#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import sys
import os

from timeit import default_timer as timer


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, drop_last=True)

    def train(self, net):
        net.train()
        # train and update
        if self.args.optimizer == "SGD":
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay = self.args.weight_decay)
        elif self.args.optimizer == "Adam":
            optimizer = torch.optim.Adam(net.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay, amsgrad = True)
        else:
            print("Invalid optimizer")

        epoch_loss = []

        start =timer()
        for iter in range(self.args.local_ep):
            print(f"local_iter: {iter}, local ep: {self.args.local_ep}, time: {timer() - start}")
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
                if self.args.verbose and (batch_idx + 1) % self.args.train_acc_batches == 0:
                    thresholds = []
                    for value in net.module.threshold.values():
                        thresholds = thresholds + [round(value.item(), 2)]
                    print('Epoch: {}, batch {}, threshold {}, leak {}, timesteps {}'.format(iter, batch_idx + 1, thresholds, net.module.leak.item(), net.module.timesteps))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)