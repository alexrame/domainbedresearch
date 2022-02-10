# This script was first copied from https://github.com/facebookresearch/InvariantRiskMinimization/blob/master/code/colored_mnist/main.py
# under the license Copyright (c) Facebook, Inc. and its affiliates.
#
# We included our new regularization Fishr. To do so:
# 1. first, we compute gradient variances on each domain (see compute_grad_variance method) using the BackPACK package
# 2. then, we compute the l2 distance between these gradient variances (see l2_between_grad_variance method)

import random
import argparse
import numpy as np

import torch
from torchvision import datasets
from torch import nn, optim, autograd

from backpack import backpack, extend
from backpack.extensions import BatchGrad, SumGradSquared, Variance

parser = argparse.ArgumentParser(description='Colored MNIST')

# Select your algorithm
parser.add_argument(
    '--algorithm',
    type=str,
    default="fishr",
    # choices=[
    #     ## Four main methods, for Table 2 in Section 5.1
    #     'erm',  # Empirical Risk Minimization
    #     'irm',  # Invariant Risk Minimization (https://arxiv.org/abs/1907.02893)
    #     'rex',  # Out-of-Distribution Generalization via Risk Extrapolation (https://icml.cc/virtual/2021/oral/9186)
    #     'fishr',  # Our proposed Fishr
    #     ## two Fishr variants, for Table 6 in Appendix B.2.4
    #     'fishr_offdiagonal',  # Fishr but on the full covariance rather than only the diagonal
    #     'fishr_notcentered',  # Fishr but without centering the gradient variances
    #     ## Fishr on the full network
    #     'fishr_features_onlyfeatures',
    #     'fishr_features',
    #     'fishr_features_notcentered',
    #     ## cosine distance
    #     'fishr_linearincrease',
    # ]
)
# Select whether you want to apply label flipping or not:
# label_flipping_prob = 0.25 by default except in Table 5 in Appendix  B.2.3 and in the right half of Table 6 in Appendix B.2.4 where label_flipping_prob = 0
parser.add_argument('--label_flipping_prob', type=float, default=0.25)

# Following hyperparameters are directly taken from:
# https://github.com/facebookresearch/InvariantRiskMinimization/blob/master/code/colored_mnist/reproduce_paper_results.sh
# They should not be modified except in case of a new proper hyperparameter search with an external validation dataset.
# Overall, we compare all approaches using the hyperparameters optimized for IRM.
parser.add_argument('--hidden_dim', type=int, default=390)
parser.add_argument('--l2_regularizer_weight', type=float, default=0.00110794568)
parser.add_argument('--lr', type=float, default=0.0004898536566546834)
parser.add_argument('--penalty_anneal_iters', type=int, default=190)
parser.add_argument('--penalty_weight', type=float, default=91257.18613115903)
parser.add_argument('--steps', type=int, default=501)

# experimental setup
parser.add_argument('--grayscale_model', action='store_true')
parser.add_argument('--n_restarts', type=int, default=1)
parser.add_argument('--seed', type=int, default=0, help='Seed for everything')

flags = parser.parse_args()

print('Flags:')
for k, v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))

random.seed(flags.seed)
np.random.seed(flags.seed)
torch.manual_seed(flags.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

final_train_accs = []
final_test_accs = []
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

            self.classifier = extend(nn.Linear(flags.hidden_dim, 1))
            for lin in [lin1, lin2, self.classifier]:
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
            self._main = extend(nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True)))
            self.fullnetwork = extend(
                nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), self.classifier)
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

    def mmd(x, y):
        if flags.algorithm == "gaussian":
            raise ValueError("gaussian")
            # Kxx = gaussian_kernel(x, x).mean()
            # Kyy = gaussian_kernel(y, y).mean()
            # Kxy = gaussian_kernel(x, y).mean()
            # return Kxx + Kyy - 2 * Kxy

        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        if "notcentered" not in flags.algorithm.split("_"):
            cent_x = x - mean_x
            cent_y = y - mean_y
        else:
            cent_x = x
            cent_y = y

        if "offdiagonal" in flags.algorithm.split("_"):
            # cova_x = (cent_x.t() @ cent_x) / (cent_x.size(0) * cent_x.size(1))
            # cova_y = (cent_y.t() @ cent_y) / (cent_y.size(0) * cent_y.size(1))
            cova_x = torch.einsum("na,nb->ab", cent_x, cent_x) / (cent_x.size(0) * cent_x.size(1))
            cova_y = torch.einsum("na,nb->ab", cent_y, cent_y) / (cent_y.size(0) * cent_y.size(1))
        else:
            cova_x = (cent_x).pow(2).mean(dim=0)
            cova_y = (cent_y).pow(2).mean(dim=0)

        mean_diff = (mean_x - mean_y).pow(2).sum()#.mean()
        cova_diff = (cova_x - cova_y).pow(2).sum()#.mean()

        if "mean" in flags.algorithm.split("_"):
            return cova_diff + mean_diff
        return cova_diff

    def mean_nll(logits, y):
        return nn.functional.binary_cross_entropy_with_logits(logits, y)

    def mean_accuracy(logits, y):
        preds = (logits > 0.).float()
        return ((preds - y).abs() < 1e-2).float().mean()

    def compute_irm_penalty(logits, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss = mean_nll(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    bce_extended = extend(nn.BCEWithLogitsLoss(reduction='sum'))

    def compute_grad_variance(input, labels, network):
        """
        Main Fishr method that computes the gradient variances in the classifier using the BackPACK package.
        """
        logits = network(input)
        loss = bce_extended(logits, labels)
        # calling first-order derivatives in the network while maintaining the per-sample gradients

        with backpack(Variance(), SumGradSquared()):
            loss.backward(inputs=list(network.parameters()), retain_graph=True, create_graph=True)

        dict_grads_variance_backpack = {
            name: (
                weights.variance.clone().view(-1)
                if "notcentered" not in flags.algorithm.split("_") else
                weights.sum_grad_squared.clone().view(-1) / input.size(0)
            ) for name, weights in network.named_parameters() if (
                "onlyfeatures" not in flags.algorithm.split("_") or
                name not in ["4.weight", "4.bias"]
            )
        }

        return dict_grads_variance_backpack

    def l2_between_grad_variance(cov_1, cov_2):
        assert len(cov_1) == len(cov_2)
        cov_1_values = [cov_1[key] for key in sorted(cov_1.keys())]
        cov_2_values = [cov_2[key] for key in sorted(cov_1.keys())]
        return (
            torch.cat(tuple([t.view(-1) for t in cov_1_values])) -
            torch.cat(tuple([t.view(-1) for t in cov_2_values]))
        ).pow(2).sum()

    # Train loop

    def pretty_print(*values):
        col_width = 13

        def format_val(v):
            if not isinstance(v, str):
                v = np.array2string(v, precision=5, floatmode='fixed')
            return v.ljust(col_width)

        str_values = [format_val(v) for v in values]
        print("   ".join(str_values))

    optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)

    pretty_print(
        'step', 'train nll', 'train acc', 'fishr penalty', 'rex penalty', 'irm penalty',
        "coral_penalty", 'test acc', "gray test acc"
    )
    for step in range(flags.steps):
        for edx, env in enumerate(envs):
            features, logits = mlp(env['images'])
            env['nll'] = mean_nll(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])
            if edx in [0, 1]:
                # when the dataset is in training
                optimizer.zero_grad()
                if "classcond" in flags.algorithm.split("_"):
                    env["grad_variance"] = {}
                    env['irm'] = {}
                    env['nllrex'] = {}
                    for label in [0, 1]:
                        idx = (env['labels'] == label).view(-1)
                        env['nllrex'][digit] = mean_nll(logits[idx], env['labels'][idx])
                        env['irm'][digit] = compute_irm_penalty(logits[idx], env['labels'][idx])
                        if "fishr" in flags.algorithm.split("_"):
                            if "features" in flags.algorithm.split("_"):
                                env["grad_variance"][digit] = compute_grad_variance(
                                    mlp.prepare_input(env["images"][idx]), env['labels'][idx],
                                    mlp.fullnetwork
                                )
                            else:
                                env["grad_variance"][digit] = compute_grad_variance(
                                    features[idx], env['labels'][idx], mlp.classifier
                                )
                else:
                    env['irm'] = compute_irm_penalty(logits, env['labels'])

                    if "features" in flags.algorithm.split("_"):
                        env["grad_variance"] = compute_grad_variance(
                            mlp.prepare_input(env["images"]), env['labels'], mlp.fullnetwork
                        )
                    else:
                        env["grad_variance"] = compute_grad_variance(
                            features, env['labels'], mlp.classifier
                        )

        train_nll = 1 / 3 * envs[0]['nll'] + 2 / 3 * envs[1]['nll']
        train_acc = 1 / 3 * envs[0]['acc'] + 2 / 3 * envs[1]['acc']

        weight_norm = torch.tensor(0.).cuda()
        for w in mlp.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_nll.clone()
        loss += flags.l2_regularizer_weight * weight_norm

        if "classcond" in flags.algorithm.split("_"):
            fishr_penalty = 0
            irm_penalty = 0
            rex_penalty = 0
            for digit in [0, 1, 2, 5, 6]:
                irm_penalty += torch.stack([envs[0]['irm'][digit], envs[1]['irm'][digit]]).mean()
                rex_penalty += (
                    envs[0]['nllrex'][digit].mean() - envs[1]['nllrex'][digit].mean()
                )**2
                if "fishr" in flags.algorithm.split("_"):
                    dict_grad_variance_averaged = {
                        name: torch.stack(
                            [
                                envs[0]["grad_variance"][digit][name],
                                envs[1]["grad_variance"][digit][name]
                            ],
                            dim=0
                        ).mean(dim=0) for name in envs[0]["grad_variance"][0]
                    }
                    fishr_penalty += (
                        l2_between_grad_variance(
                            envs[0]["grad_variance"][digit], dict_grad_variance_averaged
                        ) + l2_between_grad_variance(
                            envs[1]["grad_variance"][digit], dict_grad_variance_averaged
                        )
                    )
                else:
                    fishr_penalty = torch.tensor(0)
        else:
            irm_penalty = torch.stack([envs[0]['irm'], envs[1]['irm']]).mean()
            rex_penalty = (envs[0]['nll'].mean() - envs[1]['nll'].mean())**2
            dict_grad_variance_averaged = {
                name: torch.stack(
                    [envs[0]["grad_variance"][name], envs[1]["grad_variance"][name]], dim=0
                ).mean(dim=0) for name in envs[0]["grad_variance"]
            }
            fishr_penalty = (
                l2_between_grad_variance(envs[0]["grad_variance"], dict_grad_variance_averaged) +
                l2_between_grad_variance(envs[1]["grad_variance"], dict_grad_variance_averaged)
            )

        if flags.algorithm.startswith("erm"):
            pass
        else:
            # apply the selected regularization
            if flags.algorithm.startswith("fishr"):
                train_penalty = fishr_penalty
            elif flags.algorithm.startswith("rex"):
                train_penalty = rex_penalty
            elif flags.algorithm.startswith("irm"):
                train_penalty = irm_penalty
            else:
                raise ValueError(flags.algorithm)
            penalty_weight = (flags.penalty_weight if step >= flags.penalty_anneal_iters else 1.0)
            loss += penalty_weight * train_penalty
            if penalty_weight > 1.0:
                # Rescale the entire loss to keep backpropagated gradients in a reasonable range
                loss /= penalty_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        test_acc = envs[2]['acc']
        grayscale_test_acc = envs[3]['acc']
        if step % 20 == 0:
            pretty_print(
                np.int32(step),
                train_nll.detach().cpu().numpy(),
                envs[0]['acc'].detach().cpu().numpy(),
                envs[1]['acc'].detach().cpu().numpy(),
                fishr_penalty.detach().cpu().numpy(),
                rex_penalty.detach().cpu().numpy(),
                irm_penalty.detach().cpu().numpy(),
                envs[2]['acc'].detach().cpu().numpy(),
                envs[3]['acc'].detach().cpu().numpy(),
                envs[4]['acc'].detach().cpu().numpy(),
            )

    final_train_accs.append(train_acc.detach().cpu().numpy())
    final_test_accs.append(envs[2]['acc'].detach().cpu().numpy())

    print('Final train acc (mean/std across restarts so far):')
    print(np.mean(final_train_accs), np.std(final_train_accs))
    print('Final test acc (mean/std across restarts so far):')
    print(np.mean(final_test_accs), np.std(final_test_accs))
