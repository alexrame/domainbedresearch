from collections import OrderedDict
from backpack.extensions import BatchDiagHessian
from backpack import backpack, extend
from torch import nn


def hessian_diag(features, labels, classifier):
    # due to our reliance on backpack and DiagHessian
    logits = classifier(features.detach())

    ce_extended = extend(nn.CrossEntropyLoss(reduction='sum'))
    loss = ce_extended(logits, labels.long())
    with backpack(BatchDiagHessian()):
        loss.backward()

    dict_batchhessian = OrderedDict(
        {n: p.diag_h_batch.clone().view(p.diag_h_batch.size(0), -1) for n, p in classifier.named_parameters()}
    )
    dict_hessian = {}
    for n, _batchhessian in dict_batchhessian.items():
        # batchhessian = _batchhessian * labels.size(0)  # multiply by batch size
        dict_hessian[n] = _batchhessian.mean(dim=0)
    return dict_hessian


def hessian_diag_full(inputs, labels, network):
    logits = network(inputs)
    ce_extended = extend(nn.CrossEntropyLoss(reduction='sum'))
    loss = ce_extended(logits, labels.long())
    with backpack(BatchDiagHessian()):
        loss.backward()

    dict_batchhessian = OrderedDict(
        {n: p.diag_h_batch.clone().view(p.diag_h_batch.size(0), -1) for n, p in network.named_parameters()}
    )
    dict_hessian = {}
    for n, _batchhessian in dict_batchhessian.items():
        # batchhessian = _batchhessian * labels.size(0)  # multiply by batch size
        dict_hessian[n] = _batchhessian.mean(dim=0)
    return dict_hessian