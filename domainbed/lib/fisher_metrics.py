"""
Copy from https://github.com/tudor-berariu/fisher-information-matrix/blob/master/fisher_metrics.py

Inpired from:
https://github.com/bioinf-jku/TTUR/blob/master/fid.py
"""

import os
from typing import Dict
from domainbed.lib.misc import pdb
from collections import OrderedDict
import torch
import torch.nn.functional as Functional
from torch import Tensor
from torch import nn


def unit_trace_diag(fim: Dict[str, Tensor]) -> Dict[str, Tensor]:
    trace = sum(t.sum() for t in fim.values())
    return {n: t / trace for (n, t) in fim.items()}


# def unit_trace_diag_(fim: Dict[str, Tensor]) -> None:
#     trace = sum(t.sum() for t in fim.values()).detach_()
#     for t in fim.values():
#         t.div_(trace)


def l2norm(fim_1):
    dict_zeros = {key: torch.zeros_like(value) for key, value in fim_1.items()}
    return l2(fim_1, dict_zeros, reduce="mean")


def l2(fim_1: Dict[str, Tensor], fim_2: Dict[str, Tensor], reduce="sum"):
    assert len(fim_1) == len(fim_2)
    fim_1_values = [fim_1[key] for key in sorted(fim_1.keys())]
    fim_2_values = [fim_2[key] for key in sorted(fim_2.keys())]
    l2norm = (
        torch.cat(tuple([t.view(-1) for t in fim_1_values])) -
        torch.cat(tuple([t.view(-1) for t in fim_2_values]))
    ).pow(2)
    if reduce == "sum":
        return l2norm.sum()
    elif reduce == "mean":
        return l2norm.mean()


def frechet_diags(fim_1: Dict[str, Tensor], fim_2: Dict[str, Tensor]):
    assert len(fim_1) == len(fim_2)
    frechet_norm = 0
    for name, values1 in fim_1.items():
        values2 = fim_2[name]
        diff = ((values1 + 1e-8).sqrt() - (values2 + 1e-8).sqrt())
        frechet_norm += (diff * diff).sum()
    return 1 - frechet_norm / 2.0


# def frechet_diags_v2(fim_1: Dict[str, Tensor], fim_2: Dict[str, Tensor]):
#     assert len(fim_1) == len(fim_2)
#     frechet_norm = 0
#     for name, values1 in fim_1.items():
#         values2 = fim_2[name]
#         diff = torch.clamp((values1 + 1e-8)*(values2 + 1e-8), min=0).sqrt()
#         frechet_norm += diff.sum()
#     return 1 - frechet_norm / 2.0


def cosine_distance(fim_1: Dict[str, Tensor], fim_2: Dict[str, Tensor]):
    return Functional.cosine_similarity(
        torch.cat(tuple([t.view(-1) for t in fim_1.values()])),
        torch.cat(tuple([t.view(-1) for t in fim_2.values()])),
        dim=0
    )


def dot_product(fim_1: Dict[str, Tensor], fim_2: Dict[str, Tensor]):
    return torch.dot(
        torch.cat(tuple([t.view(-1) for t in fim_1.values()])),
        torch.cat(tuple([t.view(-1) for t in fim_2.values()]))
    )


def distance_between_dicts(dict1, dict2, strategy):
    if not isinstance(dict1, dict):
        dict1 = {e: v for e, v in enumerate(dict1)}
        dict2 = {e: v for e, v in enumerate(dict2)}

    if strategy == "l2sumvar":
        dictmean = {
            key: torch.stack([dict1[key], dict2[key]], dim=0).mean(dim=0)
            for key in dict1
            }
        return l2(dict1, dictmean, reduce="sum") + l2(dict2, dictmean, reduce="sum")
    elif strategy.startswith("l2mean"):
        return l2(dict1, dict2, reduce="mean")
    elif strategy.startswith("l2sum"):
        return l2(dict1, dict2, reduce="sum")
    elif strategy == "frechet":
        dict1_unit = unit_trace_diag(dict1)
        dict2_unit = unit_trace_diag(dict2)
        return -frechet_diags(dict1_unit, dict2_unit)
    elif strategy == "cosine":
        return - cosine_distance(dict1, dict2)
    elif strategy == "dot":
        return - dot_product(dict1, dict2)
    raise ValueError("Unknown overlap strategy")

def main():
    import torch.optim as optim
    w2 = {"w": torch.rand(50).exp()}
    w2 = unit_trace_diag(w2)

    params = torch.rand(50, requires_grad=True)
    optimizer = optim.SGD([params], lr=.001, momentum=.99, nesterov=True)
    scale = 0.1
    for step in range(10000):
        optimizer.zero_grad()
        w1 = {"w": params.exp()}
        w1 = unit_trace_diag(w1)
        overlap = frechet_diags(w1, w2)
        cos = cosine_distance(w1, w2)
        dot = dot_product(w1, w2)

        loss = -overlap * scale

        loss.backward()
        optimizer.step()
        if (step + 1) % 25 == 0:
            print("Step", step, ":", overlap.item(), cos.item(), dot.item())
    print(w1)
    print(unit_trace_diag(w2))


from backpack import backpack, extend
from backpack.extensions import BatchGrad
loss_backpack = extend(nn.BCEWithLogitsLoss())


def fisher_info(features, y, classifier, strategy="centered"):
    if not "cond" in strategy:
        return {"non_conditional": _fisher_info(features, y, classifier, strategy=strategy)}

    set_possible_y = set(list(y.cpu().detach().numpy().squeeze()))
    fisher_dict = {}
    for possible_y in set_possible_y:
        filtered_features = features[y.squeeze() == possible_y]
        filtered_y = y[y.squeeze() == possible_y]
        fisher_dict[possible_y] = _fisher_info(
            filtered_features,
            filtered_y,
            classifier,
            strategy=strategy
        )
    return fisher_dict


def _fisher_info(features, y, classifier, strategy="centered"):
    logits = classifier(features)
    tensortoderive = loss_backpack(logits, y)
    with backpack(BatchGrad()):
        tensortoderive.backward(
            inputs=list(classifier.parameters()), retain_graph=True, create_graph=True
        )

    all_fmap = OrderedDict(
        {
            n: p.grad_batch.clone().view(p.grad_batch.size(0), -1)
            for n, p in classifier.named_parameters()
        }
    )  # num_params => bsize
    dict_grads_mean = {}
    dict_grads_cov = {}
    batch_size = len(y)
    for n, fmap in all_fmap.items():
        env_grads = fmap * batch_size
        dict_grads_mean[n] = env_grads.mean(dim=0, keepdim=True)
        if strategy.startswith("centered_detach"):
            env_grads = env_grads - dict_grads_mean[n].detach()
        elif strategy.startswith("centered"):
            env_grads = env_grads - dict_grads_mean[n]
        elif strategy.startswith("nocentered"):
            pass
        else:
            raise ValueError(strategy)

        if "square" in strategy:
            covariance_env_grads = torch.einsum(
                "na,nb->ab", env_grads, env_grads) / (env_grads.size(0) - 1)
            env_grads = covariance_env_grads / env_grads.size(1)

        dict_grads_cov[n] = (env_grads).pow(2).mean(dim=0)

        if "sign" in strategy:
            dict_grads_cov[n] *= torch.sign(dict_grads_mean[n]).reshape(-1)

    return {"mean": dict_grads_mean, "cov": dict_grads_cov}



if __name__ == "__main__":
    main()
