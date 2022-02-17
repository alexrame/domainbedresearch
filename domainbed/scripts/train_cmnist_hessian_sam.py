# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pdb
import random
import argparse
import numpy as np
from collections import OrderedDict

import torch
from torchvision import datasets
from torch import nn, optim, autograd
from domainbed.lib import sam
from backpack import backpack, extend
from backpack.extensions import BatchGrad, DiagHessian, BatchDiagHessian
from backpack.extensions import SumGradSquared, Variance

parser = argparse.ArgumentParser(description='Colored MNIST')

# select your algorithm
parser.add_argument(
    '--algorithm',
    type=str,
    default="fishr",
    choices=['erm', 'irm', "rex", 'fishr', "fishr_alllayers"]
)
parser.add_argument(
    '--verbose',
    type=str,
    default="hessian",
)
parser.add_argument('--output_size', type=int, default=2, choices=[1, 2])
# hyperparameters taken from from https://github.com/facebookresearch/InvariantRiskMinimization/blob/master/code/colored_mnist/reproduce_paper_results.sh
parser.add_argument('--hidden_dim', type=int, default=390)
parser.add_argument('--l2_regularizer_weight', type=float, default=0.00110794568)
parser.add_argument('--lr', type=float, default=0.0004898536566546834)
parser.add_argument('--penalty_anneal_iters', type=int, default=190)
parser.add_argument('--penalty_weight', type=float, default=91257.18613115903)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--phosam', type=float, default=0)

# experimental setup
parser.add_argument('--grayscale_model', action='store_true')
parser.add_argument('--n_restarts', type=int, default=10)
parser.add_argument('--label_flipping_prob', type=float, default=0.25)
parser.add_argument('--seed', type=int, default=0, help='Seed for everything')
parser.add_argument('--plot', type=int, default=True)

flags = parser.parse_args()

print('Flags:')
for k, v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))

random.seed(flags.seed)
np.random.seed(flags.seed)
torch.manual_seed(flags.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

all_train_nlls = -1 * np.ones((flags.n_restarts, flags.steps))
all_train_nlls_0 = -1 * np.ones((flags.n_restarts, flags.steps))
all_train_nlls_1 = -1 * np.ones((flags.n_restarts, flags.steps))
all_test_nlls = -1 * np.ones((flags.n_restarts, flags.steps))
all_train_accs = -1 * np.ones((flags.n_restarts, flags.steps))
all_test_accs = -1 * np.ones((flags.n_restarts, flags.steps))
all_rex_penalties = -1 * np.ones((flags.n_restarts, flags.steps))
all_fishr_penalties = -1 * np.ones((flags.n_restarts, flags.steps))
all_fishr_intradomain_penalties = -1 * np.ones((flags.n_restarts, flags.steps))
all_fishr_norm = -1 * np.ones((flags.n_restarts, flags.steps))
all_hessian_penalties = -1 * np.ones((flags.n_restarts, flags.steps))
all_hessian_intradomain_penalties = -1 * np.ones((flags.n_restarts, flags.steps))
all_hessian_norm = -1 * np.ones((flags.n_restarts, flags.steps))
all_hessian_meannorm = -1 * np.ones((flags.n_restarts, flags.steps))

final_train_accs = []
final_test_accs = []
final_graytest_accs = []
for restart in range(flags.n_restarts):
    print("Restart", restart)

    # Load MNIST, make train/val splits, and shuffle train set examples

    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    mnist_val = (mnist.data[50000:], mnist.targets[50000:])

    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())

    # Build environments


    def make_environment(images, labels, e, grayscale=False):

        def torch_bernoulli(p, size):
            return (torch.rand(size) < p).float()

        def torch_xor(a, b):
            return (a - b).abs()  # Assumes both inputs are either 0 or 1

        # 2x subsample for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit; flip label with probability 0.25
        labels = (labels < 5).float()
        labels = torch_xor(labels, torch_bernoulli(flags.label_flipping_prob, len(labels)))
        # Assign a color based on the label; flip the color with probability e
        colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
        # Apply the color to the image by zeroing out the other color channel
        images = torch.stack([images, images], dim=1)
        if not grayscale:
            images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0
        return {'images': (images.float() / 255.).cuda(), 'labels': labels[:, None].cuda()}

    envs = [
        make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2),
        make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
        make_environment(mnist_val[0], mnist_val[1], 0.9),
        make_environment(mnist_val[0], mnist_val[1], 0.9, grayscale=True)
    ]

    # Define and instantiate the model


    class MLP(nn.Module):

        def __init__(self):
            super(MLP, self).__init__()
            if flags.grayscale_model:
                lin1 = nn.Linear(14 * 14, flags.hidden_dim)
            else:
                lin1 = nn.Linear(2 * 14 * 14, flags.hidden_dim)
            lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
            self.classifier = extend(nn.Linear(flags.hidden_dim, flags.output_size))
            for lin in [lin1, lin2, self.classifier]:
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
            self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True))

            self.alllayers = extend(
                nn.Sequential(lin1, nn.ReLU(False), lin2, nn.ReLU(False), self.classifier)
            )

        @staticmethod
        def prepare_input(input):
            if flags.grayscale_model:
                out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
            else:
                out = input.view(input.shape[0], 2 * 14 * 14)
            return out

        def forward(self, input):
            out = self.prepare_input(input)
            features = self._main(out)
            logits = self.classifier(features)
            return features, logits

    mlp = MLP().cuda()

    # Define loss function helpers


    def mean_nll(logits, y):
        if flags.output_size == 2:
            return nn.CrossEntropyLoss()(logits, y[:, 0].long())
        else:
            return nn.functional.binary_cross_entropy_with_logits(logits, y.float())

    def mean_accuracy(logits, y):
        if flags.output_size == 2:
            preds = (logits[:, 1] > logits[:, 0]).float().view(-1, 1)
        else:
            preds = (logits > 0.).float()
        acc = ((preds - y).abs() < 1e-2).float().mean()
        return acc

    def compute_irm_penalty(logits, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss = mean_nll(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    bce_extended = extend(nn.BCEWithLogitsLoss(reduction='sum'))
    ce_extended = extend(nn.CrossEntropyLoss(reduction='sum'))

    def compute_hessian(features, labels, classifier):
        # due to our reliance on backpack and DiagHessian
        logits = classifier(features.detach())
        assert flags.output_size == 2
        loss = ce_extended(logits, labels[:, 0].long())
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

    def compute_grads_cov(input, labels, network):
        """
        Main Fishr method that computes the gradient variances in the classifier using the BackPACK package.
        """
        logits = network(input)
        if flags.output_size == 2:
            loss = ce_extended(logits, labels[:, 0].long())
        else:
            loss = bce_extended(logits, labels.float())
        # calling first-order derivatives in the network while maintaining the per-sample gradients

        with backpack(Variance()):
            loss.backward(inputs=list(network.parameters()), retain_graph=True, create_graph=True)

        dict_grads_variance_backpack = {
            name: weights.variance.clone().view(-1) for name, weights in network.named_parameters()
        }

        return dict_grads_variance_backpack

    def l2(cov_1, cov_2):
        assert len(cov_1) == len(cov_2)
        cov_1_values = [cov_1[key] for key in sorted(cov_1.keys())]
        cov_2_values = [cov_2[key] for key in sorted(cov_2.keys())]
        return (
            torch.cat(tuple([t.view(-1) for t in cov_1_values])) -
            torch.cat(tuple([t.view(-1) for t in cov_2_values]))
        ).pow(2).sum()

    def cosine_distance(fim_1, fim_2):
        return nn.functional.cosine_similarity(
            torch.cat(tuple([t.view(-1) for t in fim_1.values()])),
            torch.cat(tuple([t.view(-1) for t in fim_2.values()])),
            dim=0
        )

    # Train loop

    def pretty_print(*values):
        col_width = 13

        def format_val(v):
            if not isinstance(v, str):
                v = np.array2string(v, precision=5, floatmode='fixed')
            return v.ljust(col_width)

        str_values = [format_val(v) for v in values]
        print("   ".join(str_values))

    if flags.phosam == 0.:
        optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)
    else:
        optimizer = sam.SAM(
            mlp.parameters(),
            optim.Adam,
            adaptive=True,
            rho=flags.phosam,
            lr=flags.lr,
        )

    pretty_print(
        'step',
        'trainnll',
        'trainacc',
        'fishrpenalty',
        "fishrnorm",
        "fishrintradomain",
        'rexpenalty',
        'irmpenalty',
        'testacc',
        "graytest acc",
        "hessian",
        "hessiannorm",
        "hessianmeannorm",
        "hessianintradomain",
    )
    for step in range(flags.steps):

        def process(compute_test=True):
            for edx, env in enumerate(envs):
                if edx in [0, 1] or compute_test:
                    features, logits = mlp(env['images'])
                else:
                    continue
                env['nll'] = mean_nll(logits, env['labels'])
                env['acc'] = mean_accuracy(logits, env['labels'])
                if edx in [0, 1]:
                    optimizer.zero_grad()
                    if "alllayers" not in flags.algorithm.split("_"):
                        env["grads_cov"] = compute_grads_cov(
                            features, env['labels'], mlp.classifier
                        )
                        if flags.verbose:
                            env["hessian"] = compute_hessian(
                                features, env['labels'], mlp.classifier
                            )
                    else:
                        env["grads_cov"] = compute_grads_cov(
                            mlp.prepare_input(env["images"]), env['labels'], mlp.alllayers
                        )
                        if flags.verbose:
                            num_hessian = int(flags.verbose.split("-")[1])
                            env["hessian"] = compute_hessian(
                                mlp.prepare_input(env["images"][:num_hessian]),
                                env['labels'][:num_hessian], mlp.alllayers
                            )

            train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
            train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()

            weight_norm = torch.tensor(0.).cuda()
            for w in mlp.parameters():
                weight_norm += w.norm().pow(2)

            loss = train_nll.clone()
            loss += flags.l2_regularizer_weight * weight_norm

            # irm_penalty = torch.stack([envs[0]['irm'], envs[1]['irm']]).mean()
            irm_penalty = torch.zeros_like(loss)
            rex_penalty = (envs[0]['nll'].mean() - envs[1]['nll'].mean())**2
            dict_grads_cov_mean = {
                key: torch.stack([envs[0]["grads_cov"][key], envs[1]["grads_cov"][key]],
                                 dim=0).mean(dim=0) for key in envs[0]["grads_cov"]
            }
            fishr_penalty_tomean = (
                l2(envs[0]["grads_cov"], dict_grads_cov_mean) +
                l2(envs[1]["grads_cov"], dict_grads_cov_mean)
            )
            fishr_penalty = (l2(envs[0]["grads_cov"], envs[1]["grads_cov"]))
            fishr_intradomain_penalty = cosine_distance(envs[0]["grads_cov"], envs[1]["grads_cov"])
            fishr_zero = {
                key: torch.zeros_like(value) for key, value in dict_grads_cov_mean.items()
            }
            fishr_norm = (l2(envs[0]["grads_cov"], fishr_zero) - l2(envs[1]["grads_cov"], fishr_zero))**2

            if flags.verbose:
                hessian_penalty = (l2(envs[0]["hessian"], envs[1]["hessian"]))
                dict_hessian_zero = {
                    key: torch.zeros_like(value) for key, value in envs[0]["hessian"].items()
                }
                hessian_norm = (l2(envs[0]["hessian"], dict_hessian_zero) - l2(envs[1]["hessian"], dict_hessian_zero))**2
                hessian_meannorm = (l2(envs[0]["hessian"], dict_hessian_zero) + l2(envs[1]["hessian"], dict_hessian_zero)) / 2
                hessian_intradomain_penalty = cosine_distance(envs[0]["hessian"], envs[1]["hessian"])

            # apply the good regularization
            if flags.algorithm == "erm":
                pass
            else:
                if "fishr" in flags.algorithm:
                    train_penalty = fishr_penalty_tomean
                elif flags.algorithm == "rex":
                    train_penalty = rex_penalty
                elif flags.algorithm == "irm":
                    train_penalty = irm_penalty
                else:
                    raise ValueError(flags.algorithm)
                penalty_weight = (
                    flags.penalty_weight if step >= flags.penalty_anneal_iters else 1.0
                )
                loss += penalty_weight * train_penalty
                if penalty_weight > 1.0:
                    # Rescale the entire loss to keep gradients in a reasonable range
                    loss /= penalty_weight

            dict_output = {
                "loss": loss,
                "train_nll": train_nll,
                "train_acc": train_acc,
                "fishr_penalty": fishr_penalty,
                "fishr_norm": fishr_norm,
                "fishr_intradomain_penalty": fishr_intradomain_penalty,
                "rex_penalty": rex_penalty,
                "irm_penalty": irm_penalty,
                "hessian_penalty": hessian_penalty,
                "hessian_norm": hessian_norm,
                "hessian_meannorm": hessian_meannorm,
                "hessian_intradomain_penalty": hessian_intradomain_penalty
            }
            return dict_output

        if flags.phosam == 0.:
            first_step = process()
            optimizer.zero_grad()
            first_step["loss"].backward()
            optimizer.step()
        else:
            # first forward-backward pass
            first_step = process()
            optimizer.zero_grad()
            first_step["loss"].backward()
            optimizer.first_step()

            # second forward-backward pass
            optimizer.zero_grad()
            second_step = process(compute_test=False)
            optimizer.zero_grad()
            # make sure to do a full forward pass
            second_step["loss"].backward()
            optimizer.second_step()

        test_acc = envs[2]['acc']
        grayscale_test_acc = envs[3]['acc']
        if step % 100 == 0 or flags.penalty_anneal_iters - 15 < step < flags.penalty_anneal_iters + 15:
            pretty_print(
                np.int32(step),
                first_step["train_nll"].detach().cpu().numpy(),
                first_step["train_acc"].detach().cpu().numpy(),
                first_step["fishr_penalty"].detach().cpu().numpy(),
                first_step["fishr_norm"].detach().cpu().numpy(),
                first_step["fishr_intradomain_penalty"].detach().cpu().numpy(),
                first_step["rex_penalty"].detach().cpu().numpy(),
                first_step["irm_penalty"].detach().cpu().numpy(),
                test_acc.detach().cpu().numpy(),
                grayscale_test_acc.detach().cpu().numpy(),
                first_step["hessian_penalty"].detach().cpu().numpy() if flags.verbose else "",
                first_step["hessian_norm"].detach().cpu().numpy() if flags.verbose else "",
                first_step["hessian_meannorm"].detach().cpu().numpy() if flags.verbose else "",
                first_step["hessian_intradomain_penalty"].detach().cpu().numpy() if flags.verbose else "",
            )
        if flags.plot:
            all_train_nlls[restart, step] = first_step["train_nll"].detach().cpu().numpy()
            all_train_nlls_0[restart, step] = envs[0]['nll'].mean().detach().cpu().numpy()
            all_train_nlls_1[restart, step] = envs[1]['nll'].mean().detach().cpu().numpy()
            all_test_nlls[restart, step] = envs[2]['nll'].mean().detach().cpu().numpy()
            all_train_accs[restart, step] = first_step["train_acc"].detach().cpu().numpy()
            all_test_accs[restart, step] = envs[2]['acc'].mean().detach().cpu().numpy()
            all_rex_penalties[restart, step] = first_step["rex_penalty"].detach().cpu().numpy()
            all_fishr_penalties[restart, step] = first_step["fishr_penalty"].detach().cpu().numpy()
            all_fishr_norm[restart, step] = first_step["fishr_norm"].detach().cpu().numpy()
            all_fishr_intradomain_penalties[restart, step] = first_step["fishr_intradomain_penalty"].detach().cpu().numpy()
            if flags.verbose:
                all_hessian_penalties[restart, step] = first_step["hessian_penalty"].detach().cpu().numpy()
                all_hessian_norm[restart, step] = first_step["hessian_norm"].detach().cpu().numpy()
                all_hessian_meannorm[restart, step] = first_step["hessian_meannorm"].detach().cpu().numpy()
                all_hessian_intradomain_penalties[restart, step] = \
                    first_step["hessian_intradomain_penalty"].detach().cpu().numpy()

    final_train_accs.append(first_step["train_acc"].detach().cpu().numpy())
    final_test_accs.append(test_acc.detach().cpu().numpy())
    final_graytest_accs.append(grayscale_test_acc.detach().cpu().numpy())
    print('Final train acc (mean/std across restarts so far):')
    print(np.mean(final_train_accs), np.std(final_train_accs))
    print('Final test acc (mean/std across restarts so far):')
    print(np.mean(final_test_accs), np.std(final_test_accs))
    print('Final gray test acc (mean/std across restarts so far):')
    print(np.mean(final_graytest_accs), np.std(final_graytest_accs))

if flags.plot:
    # plot_x = np.linspace(0, flags.steps, flags.steps)
    # from pylab import *

    # figure()
    # xlabel('epoch')
    # ylabel('loss')
    # title('Train penalties')
    # # plot(plot_x, all_train_nlls.mean(0) * 0.1, ls="dotted", label='train_nll')
    # plot(plot_x, all_rex_penalties.mean(0), label='rex_penalty')
    # plot(plot_x, all_fishr_penalties.mean(0), ls="--", label='fishr_penalty')
    # plot(plot_x, all_hessian_penalties.mean(0), ls="dashdot", label='hessian_penalty')
    # legend(prop={'size': 11}, loc="upper right")
    # savefig('penalties.pdf')

    # figure()
    # title('Gradient and Hessian norms')
    # plot(plot_x, all_fishr_norm.mean(0), ls="--", label='fishr_norm')
    # plot(plot_x, all_hessian_norm.mean(0), ls="dashdot", label='hessian_norm')
    # yscale('log')
    # legend(prop={'size': 11}, loc="upper right")
    # savefig('norms.pdf')

    import os
    directory = "np_arrays_paper" + "_" + flags.algorithm + "_" + flags.verbose + "_" + str(
        flags.output_size
    ) + "_" + str(flags.phosize)
    if not os.path.exists(directory):
        os.makedirs(directory)

    outfile = "all_train_nlls"
    np.save(directory + "/" + outfile, all_train_nlls)

    outfile = "all_train_nlls_0"
    np.save(directory + "/" + outfile, all_train_nlls_0)

    outfile = "all_train_nlls_1"
    np.save(directory + "/" + outfile, all_train_nlls_1)

    outfile = "all_test_nlls"
    np.save(directory + "/" + outfile, all_test_nlls)
    outfile = "all_test_accs"
    np.save(directory + "/" + outfile, all_test_accs)
    outfile = "all_train_accs"
    np.save(directory + "/" + outfile, all_train_accs)
    outfile = "all_fishr_penalties"
    np.save(directory + "/" + outfile, all_fishr_penalties)

    outfile = "all_fishr_intradomain_penalties"
    np.save(directory + "/" + outfile, all_fishr_intradomain_penalties)

    outfile = "all_rex_penalties"
    np.save(directory + "/" + outfile, all_rex_penalties)

    if flags.verbose:
        outfile = "all_hessian_penalties"
        np.save(directory + "/" + outfile, all_hessian_penalties)

        outfile = "all_fishr_norm"
        np.save(directory + "/" + outfile, all_fishr_norm)

        outfile = "all_hessian_norm"
        np.save(directory + "/" + outfile, all_hessian_norm)
        outfile = "all_hessian_meannorm"
        np.save(directory + "/" + outfile, all_hessian_meannorm)

        outfile = "all_hessian_intradomain_penalties"
        np.save(directory + "/" + outfile, all_hessian_intradomain_penalties)

if 1 == 0:
    import matplotlib.pyplot as plt
    fontsize = "x-large"
    colors = [plt.get_cmap("Dark2")(i / float(4)) for i in range(4)]
    plot_x = np.linspace(0, 501, 501)

    fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(6, 4))

    #trainnnl = ax1.plot(
    #plot_x, all_train_nlls.mean(0), "--", color=colors[0], label="Train nll", markersize=5
    #)[0]
    #rex = ax1.plot(
    #    plot_x, all_rex_penalties.mean(0), "-", color=colors[0], label="V-REx", markersize=5
    #)[0]
    ax1.plot(
        plot_x,
        all_fishr_penalties.mean(0),
        "-",
        color=colors[0],
        label=r"Gradients covariance: $||\mathbf{C}_{90\%} - \mathbf{C}_{80\%}||_{F}^{2}$",
        markersize=5
    )[0]
    ax1.plot(
        plot_x,
        all_fishr_intradomain_penalties.mean(0),
        "--",
        color=colors[0],
        label=r"Gradients covariance: $||\mathbf{C}_{e}^0 - \mathbf{C}_{e}^1||_{F}^{2}$",
        markersize=5
    )[0]
    plt.xticks([0, 100, 190, 300, 400, 500])
    #ax1.set_xlim([0, 1.0])

    ax1.set_ylim(0, ymax=max(all_fishr_penalties.mean(0)))
    # plt.yticks([78, 80.0, 82, 84])

    ax1.plot(
        plot_x,
        all_hessian_penalties.mean(0) * 100,
        '-',
        color=colors[1],
        label=r"Hessian: $||\mathbf{H}_{90\%} - \mathbf{H}_{80\%}||_{F}^{2} * 100$",
        markersize=10
    )
    ax1.plot(
        plot_x,
        all_hessian_penalties.mean(0) * 100,
        '--',
        color=colors[1],
        label=r"Hessian: $||\mathbf{H}_{e}^0 - \mathbf{H}_{e}^1||_{F}^{2} * 100$",
        markersize=10
    )
    plt.vlines(x=190, ymin=-1, ymax=1, linestyles="dotted", colors=colors[2])
    ax1.set_xlabel(r"Epoch", fontsize=fontsize)
    ax1.set_ylabel(r'Distance', fontsize="x-large")
    ax1.legend(loc="lower right", bbox_to_anchor=(1.0, 0.6), fontsize=fontsize)

    plt.tight_layout()
    plt.show()
