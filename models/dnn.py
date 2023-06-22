# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sigmoid, Module
from torch.nn.init import kaiming_uniform_, xavier_uniform_

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class DNN(Module):
    def __init__(self, inputs_num):
        self.cfg = {'num_layer1_out':10,'num_layer2_out':18,'num_layer3_out':8,'num_layer4_out':1}
        super(DNN, self).__init__()
        self.layer1 = Linear(inputs_num, self.cfg['num_layer1_out'])
        kaiming_uniform_(self.layer1.weight, nonlinearity='relu')
        self.active1 = ReLU()
        self.layer2 = Linear(self.cfg['num_layer1_out'],self.cfg['num_layer2_out'])
        kaiming_uniform_(self.layer2.weight, nonlinearity='relu')
        self.active2 = ReLU()
        self.layer3 = Linear(self.cfg['num_layer2_out'],self.cfg['num_layer3_out'])
        kaiming_uniform_(self.layer3.weight, nonlinearity='relu')
        self.active3 = ReLU()
        self.layer4 = Linear(self.cfg['num_layer3_out'], self.cfg['num_layer4_out'])
        xavier_uniform_(self.layer4.weight)
        self.active4 = Sigmoid()
        #self.active4 = FocalLoss()

    def forward(self, x):
        x = self.layer1(x)
        x = self.active1(x)
        x = self.layer2(x)
        x = self.active2(x)
        x = self.layer3(x)
        x = self.active3(x)
        x = self.layer4(x)
        x = self.active4(x)
        return x