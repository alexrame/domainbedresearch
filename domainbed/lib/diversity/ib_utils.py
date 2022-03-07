import torch
from itertools import combinations
import random
import numpy as np
import os


def tanh_clip(x, clip_val):
    '''
    soft clip values to the range [-clip_val, +clip_val]
    # For clipping score s to range [−c, c], we applied the non-linearity
    # s′ = c tanh( s ), which is linear around 0 and saturates as one approaches ±c. We use c = 10
    '''
    x_clip = clip_val * torch.tanh((1. / clip_val) * x)
    return x_clip


def permute_tensor(t, permut=None, dim=0):
    len_t = t.size(dim)
    if permut is None:
        permut = np.random.permutation(len_t)
    if dim == 0:
        return t[permut]
    elif dim == 1:
        return t[:, permut]
    else:
        raise ValueError(dim)


def reparameterization_trick(enc_mean, enc_var=1, active=True):
    if active:
        latent = enc_mean + enc_var * torch.randn_like(enc_mean)
    else:
        latent = enc_mean
    return latent


def positive_couples(batch_ib, batch_classes):
    input_0 = []
    input_1 = []
    for i, j in combinations(range(batch_ib.size(1)), 2):
        input_0.append(batch_ib[:, i].reshape(-1, batch_ib.size(-1)))
        input_1.append(batch_ib[:, j].reshape(-1, batch_ib.size(-1)))
    if batch_classes is not None:
        return format_inputs(input_0, input_1), batch_classes.view(-1,)
    return format_inputs(input_0, input_1), None


def negative_couples(batch_ib):
    # batch_ib is num_domains, num_members, bsize, size
    input_0 = []
    input_1 = []
    num_members = batch_ib.size(1)
    for i, j in combinations(range(num_members), 2):
        input_0.append(batch_ib[:, i].reshape(-1, batch_ib.size(-1)))
        input_1.append(permute_tensor(batch_ib[:, j].reshape(-1, batch_ib.size(-1))))

    return format_inputs(input_0, input_1)


def negative_couples_per_domain(batch_ib):
    # batch_ib is num_domains, num_members, bsize, size
    input_0 = []
    input_1 = []
    num_members = batch_ib.size(1)
    for i, j in combinations(range(num_members), 2):
        input_0.append(batch_ib[:, i].reshape(-1, batch_ib.size(-1)))
        input_1.append(permute_tensor(batch_ib[:, j], dim=1).reshape(-1, batch_ib.size(-1)))

    return format_inputs(input_0, input_1)


def negative_couples_conditional(batch_ib, batch_classes, num_classes=None, embedding_layers=None):
    input_0 = []
    input_1 = []
    classes = []
    batch_classes_flat = batch_classes.view((-1,))
    features_size = batch_ib.size(-1)
    num_members = batch_ib.size(1)
    batch_ib_transposed = torch.transpose(batch_ib, 0, 1).reshape(
        num_members,
        batch_ib.size(0) * batch_ib.size(2), features_size
    )  # Members, domains * bs, features
    for classe in range(num_classes):
        batch_ib_filtered = batch_ib_transposed[:, batch_classes_flat == classe, :]
        for memi, memj in combinations(range(num_members), 2):
            if batch_ib_filtered.size(1) == 0:
                continue
            elif batch_ib_filtered.size(1) == 1:
                if embedding_layers is not None and os.environ.get("EMB"):
                    neg = embedding_layers[memj].weight.data[classe].clone().reshape(
                        -1, features_size
                    )
                    embedding_layers[memj].weight.data[classe] = batch_ib_filtered[memj].reshape(
                        (features_size,)
                    ).detach()
                else:
                    continue
            else:
                neg = permute_tensor(batch_ib_filtered[memj]).reshape(-1, features_size)
            input_0.append(batch_ib_filtered[memi].reshape(-1, features_size))
            input_1.append(neg)
            classes.append(classe * torch.ones(batch_ib_filtered.size(1),))

    new_classes = torch.cat(classes).view(-1,).to("cuda").long()
    return format_inputs(input_0, input_1), new_classes


def format_inputs(input_0, input_1):
    return torch.cat((torch.cat(input_0, 0), torch.cat(input_1, 0)), dim=1)
