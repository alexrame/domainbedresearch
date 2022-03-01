from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
import numpy as np



class SoftCrossEntropyLoss(nn.modules.loss._Loss):

    def forward(self, input, target):
        """
        Cross entropy that accepts soft targets
        Args:
            pred: predictions for neural network
            targets: targets, can be soft
            size_average: if false, sum is returned instead of mean
        Examples::
            input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
            input = torch.autograd.Variable(out, requires_grad=True)
            target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
            target = torch.autograd.Variable(y1)
            loss = cross_entropy(input, target)
            loss.backward()
        """
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))


def entropy_regularizer(logits):
    # Num_domains, bs, num_classes
    logits = logits.reshape(-1, logits.size(-1))
    # num_samples, num_classes
    x = torch.softmax(logits, dim=1)
    entropy = - x * torch.log(x + 1e-10)  #bs * num_classes
    return entropy.sum()/entropy.size(0)

class KLLoss(nn.Module):
    def __init__(self):

        super(KLLoss, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T=3

        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data = target_data + 10**(-7)
        target = Variable(target_data.data.cuda(), requires_grad=False)
        loss = T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss



def sigmoid_rampup(current, rampup_length, kappa=5.):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-kappa * phase * phase))


def get_current_consistency_weight(epoch, consistency_rampup, epoch_start_consistency_rampup=0, penalty_anneal_method=False):
    if epoch < epoch_start_consistency_rampup:
        return 0
    if consistency_rampup <= epoch - epoch_start_consistency_rampup:
        return 1
    if int(penalty_anneal_method) in [2, 4]:
        return 1 * (epoch - epoch_start_consistency_rampup > consistency_rampup)
    raise ValueError(penalty_anneal_method)
    if int(penalty_anneal_method) in [0, 6]:
        return (epoch - epoch_start_consistency_rampup) / consistency_rampup
    if int(penalty_anneal_method) in [1, 3]:
        return sigmoid_rampup(epoch - epoch_start_consistency_rampup, consistency_rampup)
    raise ValueError(penalty_anneal_method)
