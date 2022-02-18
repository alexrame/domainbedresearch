from collections import OrderedDict
from backpack.extensions import BatchDiagHessian, HMP
from backpack import backpack, extend
from torch import nn
import torch


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


def rademacher(shape, dtype=torch.float32, device="gpu"):
    """Sample from Rademacher distribution."""
    rand = ((torch.rand(shape) < 0.5)) * 2 - 1
    return rand.to(dtype).to(device)


def hutchinson_trace_hmp(inputs, labels, network, V=1000, V_batch=20, device="gpu"):
    """Hessian trace estimate using BackPACK's HMP extension. Perform `V_batch` Hessian multiplications at a time."""
    logits = network(inputs)
    ce_extended = extend(nn.CrossEntropyLoss(reduction='sum'))
    loss = ce_extended(logits, labels.long())
    with backpack(HMP()):
        loss.backward(retain_graph=True)
    V_count = 0
    trace = 0
    while V_count < V:
        V_missing = V - V_count
        V_next = min(V_batch, V_missing)
        for param in network.parameters():
            v = rademacher((V_next, *param.shape), device=device)
            Hv = param.hmp(v).detach()
            vHv = torch.einsum("i,i->", v.flatten(), Hv.flatten().detach())
            trace += vHv / V
        V_count += V_next

    return trace
