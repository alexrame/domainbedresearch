# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Things that don't belong anywhere else
"""

import hashlib
import sys
from collections import OrderedDict, defaultdict
from numbers import Number
import operator
import numpy as np
import torch
import json
import torch.nn as nn
from torch.utils.data import dataset
from torch.utils.data.dataset import Dataset
from collections import Counter
import socket
import copy
import torch.nn.functional as F
import random
try:
    from pyhessian import hessian
except:
    hessian = None

try:
    from torchmetrics import Precision, Recall
except:
    Precision, Recall = None, None


def is_dumpable(value):
    try:
        json.dumps(value)
    except:
        return False
    return True


def get_featurizer(network):
    return nn.Sequential(*list(network.children())[:-1])


def compute_hessian(network, loader, maxIter=100):
    if hessian is None:
        return 0

    hessian_comp_soup = hessian(
        network, nn.CrossEntropyLoss(reduction='mean'), dataloader=loader, cuda=True
    )
    return np.mean(hessian_comp_soup.trace(maxIter=maxIter))


class SWA():

    def __init__(self, network, hparams, swa_start_iter=100, global_iter=0):
        self.network = network
        self.network_swa = copy.deepcopy(network)
        self.network_swa.eval()

        self.global_iter = global_iter
        self.layerwise = hparams.get("layerwise", "")
        self.hparams = hparams
        self.swa_start_iter = swa_start_iter
        if self.layerwise:
            self.list_layers_count = [0 for _ in self.network.parameters()]
        else:
            self.swa_count = 0

    def update(self):
        self.global_iter += 1
        if self.global_iter >= self.swa_start_iter:
            if self.layerwise:
                self._update_layerwise()
            else:
                self._update_all()
        elif True:
            for param_q, param_k in zip(self.network.parameters(), self.network_swa.parameters()):
                param_k.data = param_q.data
        return self.compute_distance_nets()

    def compute_distance_nets(self):
        dist_l2 = 0
        cos = 0
        count_params = 0
        for param_q, param_k in zip(self.network.parameters(), self.network_swa.parameters()):
            dist_l2 += (param_k.data.reshape(-1) - param_q.data.reshape(-1)).pow(2).sum()
            num_params = int(param_q.numel())
            count_params += num_params
            cos += (param_k * param_q).sum() / (param_k.norm() * param_q.norm()) * num_params
        return {"swa_l2": dist_l2 / count_params, "swa_cos": cos / count_params}

    def _update_layerwise(self):
        layerwise_split = self.layerwise.split("-") if isinstance(self.layerwise, str) else []
        for i, (param_q, param_k) in enumerate(
            zip(self.network.parameters(), self.network_swa.parameters())
        ):
            if "bin" in layerwise_split:
                if random.random() < self.hparams["swa_bin"]:
                    continue
            if "exp" in layerwise_split:
                Z = np.random.exponential(scale=self.hparams["swa_exp"], size=1)[0]
            else:
                Z = 1
            count = self.list_layers_count[i]
            param_k.data = (param_k.data * count + param_q.data * Z) / (count + Z)
            self.list_layers_count[i] += Z

    def _update_all(self):
        self.swa_count += 1
        for param_q, param_k in zip(self.network.parameters(), self.network_swa.parameters()):
            param_k.data = (param_k.data * self.swa_count + param_q.data) / (1. + self.swa_count)

    def get_classifier(self):
        return list(self.network_swa.children())[-1]

    def get_featurizer(self):
        return get_featurizer(self.network_swa)


class Soup():

    def __init__(self, networks):
        self.networks = networks
        if len(networks) != 0:
            self.network_soup = copy.deepcopy(networks[0])
            self.update()
        else:
            self.network_soup = None

    def update(self):
        for param in zip(
            self.network_soup.parameters(), *[net.parameters() for net in self.networks]
        ):
            param_k = param[0]
            param_k.data = sum(param[1:]).data / len(self.networks)

    # def update_tscaled(self, temperatures):
    #     if len(self.networks) == 0:
    #         return
    #     sum_temperatures = sum(temperatures)
    #     # print(f"Weighted soup with temperatures: {temperatures} of sum: {sum_temperatures}")
    #     for param in zip(
    #         self.network_soup.named_parameters(), *[net.parameters() for net in self.networks]
    #     ):
    #         name = param[0][0]
    #         if name not in ["1.weight", "1.bias"]:
    #             continue
    #         param_k = param[0][1]
    #         weighted_sum = sum([p.data * t for p, t in zip(param[1:], temperatures)])
    #         param_k.data = weighted_sum / sum_temperatures

    def get_featurizer(self):
        return get_featurizer(self.network_soup)


def get_ece(proba_pred, accurate, n_bins=15, min_pred=0, verbose=False, **args):
    """
    Compute ECE and write to file
    """
    if min_pred == "minpred":
        min_pred = min(proba_pred)
    else:
        assert min_pred >= 0
    bin_boundaries = np.linspace(min_pred, 1., n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    acc_in_bin_list = []
    avg_confs_in_bins = []
    list_prop_bin = []
    ece = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(proba_pred > bin_lower, proba_pred <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        list_prop_bin.append(prop_in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accurate[in_bin])
            avg_confidence_in_bin = np.mean(proba_pred[in_bin])
            delta = avg_confidence_in_bin - accuracy_in_bin
            acc_in_bin_list.append(accuracy_in_bin)
            avg_confs_in_bins.append(avg_confidence_in_bin)
            ece += np.abs(delta) * prop_in_bin
            if verbose:
                print(
                    f"From {bin_lower:4.5} to {bin_upper:4.5} and mean {avg_confidence_in_bin:3.5}, "
                    f"{(prop_in_bin * 100):4.5} % samples with accuracy {accuracy_in_bin:4.5}"
                )
        else:
            avg_confs_in_bins.append(None)
            acc_in_bin_list.append(None)

    return ece * 100


def apply_temperature_on_logits(logits, temperature):
    """
    Apply temperature relaxation on logits
    """
    reshaped_temperature = temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
    return logits / reshaped_temperature


def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
        torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
        torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).pow(2).mean()


class MovingAverage:

    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.ema_data = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.ema_data[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.ema_data[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data

    def update_value(self, data):
        data = data.view(1, -1)
        if self._updates == 0:
            previous_data = torch.zeros_like(data)
        else:
            previous_data = self.ema_data

        ema_data = self.ema * previous_data + (1 - self.ema) * data

        self.ema_data = ema_data.clone().detach()
        self._updates += 1

        if self._oneminusema_correction:
            return ema_data / (1 - self.ema)
        return ema_data


def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights


def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)


def print_separator():
    print("=" * 80)


def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        return
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.3f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]

    print(sep.join([format_val(x) for x in row]), end_)


def print_rows(row1, row2):
    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.3f}".format(x)
        return str(x)

    to_print = ", ".join([format_val(x1) + ": " + format_val(x2) for x1, x2 in zip(row1, row2)])
    print("{" + to_print + "}")


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""

    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys

    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]

    def __len__(self):
        return len(self.keys)


class MergeDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        super(MergeDataset, self).__init__()
        self.datasets = datasets

    def __getitem__(self, key):
        count = 0
        for d in self.datasets:
            if key - count >= len(d):
                count += len(d)
            else:
                return d[key - count]
        raise ValueError(key)

    def __len__(self):
        return sum([len(d) for d in self.datasets])


def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert (n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)


def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    n = len(iterable)
    indices = sorted(random.sample(range(n), r))
    return [iterable[i] for i in indices]


def mix_up_loss(self, x, y, x2=None, y2=None, mix_label=True):
    x1, y1 = x, y
    if x2 is None:
        idxes = torch.randperm(len(x))
        x2, y2 = x[idxes], y[idxes]
    lam = np.random.beta(self.hparams["mixup_alpha"], self.hparams["mixup_alpha"])
    x_mixed = lam * x1 + (1 - lam) * x2
    predictions = self.network(x_mixed)
    if mix_label:
        return lam * F.cross_entropy(predictions, y1) + (1 - lam) * F.cross_entropy(predictions, y2)
    else:
        return F.cross_entropy(predictions, y1)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cut_mix_loss(self, x, y, x2=None, y2=None, mix_label=True):
    x1, y1 = x, y
    if x2 is None:
        rand_index = torch.randperm(len(y))
        x2, y2 = x[rand_index], y[rand_index]
    lam = np.random.beta(self.hparams["mixup_alpha"], self.hparams["mixup_alpha"])
    bbx1, bby1, bbx2, bby2 = rand_bbox(x1.size(), lam)
    # mixed input
    x1[:, :, bbx1:bbx2, bby1:bby2] = x2[:, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x1.size()[-1] * x1.size()[-2]))
    predictions = self.network(x1)
    if mix_label:
        return lam * F.cross_entropy(predictions, y1) + (1 - lam) * F.cross_entropy(predictions, y2)
    else:
        return F.cross_entropy(predictions, y1)


def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []
    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0
        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]
        min_n = min(len(xi), len(xj))
        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))
    return pairs


def compute_correct_batch(predictions, weights, y, device):
    correct = 0
    total = 0
    weights_offset = 0

    p = predictions
    if weights is None:
        batch_weights = torch.ones(len(y))
    else:
        batch_weights = weights[weights_offset:weights_offset + len(y)]
        weights_offset += len(y)
    batch_weights = batch_weights.to(device)
    if p.size(1) == 1:
        correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
    else:
        correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
    total += batch_weights.sum().item()
    return correct, total


def accuracy(network, loader, weights, device):
    network.eval()

    corrects = defaultdict(int)  # key -> int
    totals = defaultdict(int)  # key -> int

    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            logits = network.predict(x)
            if not isinstance(logits, dict):
                logits = {"main": logits}

            for key in logits:
                correct, total = compute_correct_batch(logits[key], weights, y, device)
                if key in ["main", ""]:
                    key_name = "acc"
                else:
                    key_name = key
                corrects[key_name] += correct
                totals[key_name] += total

    results = dict()
    for key in corrects:
        results[key] = corrects[key] / totals[key]

    network.train()

    return results


class Tee:

    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)


class CustomToRegularDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        item = self.dataset[index]
        return item["x"], item["y"]

    def __len__(self) -> int:
        return len(self.dataset)


def dict_batch_to_device(batch, device):
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
        elif isinstance(batch[key], dict):
            batch[key] = dict_batch_to_device(batch[key], device)
    return batch


class DictDataset(Dataset):

    def __init__(self, dict):
        keys = list(dict.keys())
        self.keys = keys
        # assert all(dict[keys[0]].size(0) == dict[k].size(0) for k in keys), "Size mismatch between tensors"
        self.dict = dict

    def __getitem__(self, index):
        return {key: self.dict[key][index] for key in self.keys}

    def __len__(self):
        return len(self.dict[self.keys[0]])


def get_machine_name():
    return socket.gethostname()


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes).to(device=labels.device)
    return y[labels]


def count_param(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_requires_grad(module, tf=False):
    module.requires_grad = tf
    for param in module.parameters():
        param.requires_grad = tf
