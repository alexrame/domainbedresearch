# This script was first copied from https://github.com/facebookresearch/InvariantRiskMinimization/blob/master/code/colored_mnist/main.py
# under the license Copyright (c) Facebook, Inc. and its affiliates.
#
# We included our new regularization Fishr. To do so:
# 1. first, we compute gradient variances on each domain (see compute_grad_statistics method) using the BackPACK package
# 2. then, we compute the l2 distance between these gradient variances (see l2_between_dicts method)

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
    default="fishr_onlyclassifier",
    # choices=[
    #     ## Main baselines, for Table 2 in Section 4.1
    #     'erm',  # Empirical Risk Minimization
    #     'irm',  # Invariant Risk Minimization (https://arxiv.org/abs/1907.02893)
    #     'rex',  # Out-of-Distribution Generalization via Risk Extrapolation (https://icml.cc/virtual/2021/oral/9186)
    #     "coral",
    #     "coral_offdiagonal",
    #     "coral_mean",
    #     "coral_mean_offdiagonal",
    #     ## Fishr
    #     "fishr_onlyclassifier",  # only in the classifier
    #     'fishr_alllayers',
    #     'fishr_onlyextractor',
    #     ## Fishr variants, for Table 7 in Appendix B.2.4
    #     'fishr_onlyclassifier_offdiagonal',  # Fishr but on the full covariance rather than only the diagonal
    #     'fishr_onlyclassifier_notcentered',  # Fishr but without centering the gradient variances
    #     'fishr_alllayers_notcentered',
    #     'fishr_onlyextractor_notcentered',
    #     ## linear increase of the regularization weights $\lambda$ after step 500
    #     'fishr_onlyclassifier_linearincrease',
    #     'fishr_alllayers_linearincrease',
    #     'fishr_onlyextractor_linearincrease',
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

            self.classifier = extend(nn.Linear(flags.hidden_dim, 1))
            for lin in [lin1, lin2, self.classifier]:
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
            self._main = extend(nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True)))
            self.alllayers = extend(
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


    def compute_nll(logits, y):
        return nn.BCEWithLogitsLoss(reduction="none")(logits, y)

    def compute_mse(logits, y):
        return compute_nll(logits, y)

        # return nn.MSELoss(reduction="none")(torch.softmax(logits, dim=1), y)

    def mean_accuracy(logits, y):
        preds = (logits > 0.).float()
        return ((preds - y).abs() < 1e-2).float().mean()

    def compute_irm_penalty(logits, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss = compute_nll(logits * scale, y).mean()
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    bce_extended = extend(nn.BCEWithLogitsLoss(reduction='sum'))

    def compute_grad_variance(input, labels, network):
        """
        Main Fishr method that computes the gradient variances using the BackPACK package.
        """
        logits = network(input)
        loss = bce_extended(logits, labels)
        # calling first-order derivatives in the network while maintaining the per-sample gradients

        with backpack(Variance(), SumGradSquared()):
            loss.backward(inputs=list(network.parameters()), retain_graph=True, create_graph=True)

        dict_grads_variance = {
            name: (
                weights.variance.clone().view(-1)
                if "notcentered" not in flags.algorithm.split("_") else
                weights.sum_grad_squared.clone().view(-1) / input.size(0)
            ) for name, weights in network.named_parameters() if (
                "onlyextractor" not in flags.algorithm.split("_") or
                name not in ["4.weight", "4.bias"]
            )
        }

        return dict_grads_variance

    def compute_grad_covariance(features, labels, classifier):
        """
        Main Fishr method that computes the gradient covariances.
        We do this by hand from individual gradients obtained with BatchGrad from BackPACK.
        This is not possible to do so in the features extractor for memory reasons!
        Indeed, covariances would involve |\gamma|^2 components.
        """
        logits = classifier(features)
        loss = bce_extended(logits, labels)
        # calling first-order derivatives in the classifier while maintaining the per-sample gradients
        with backpack(BatchGrad()):
            loss.backward(
                inputs=list(classifier.parameters()), retain_graph=True, create_graph=True
            )

        dict_grads = {
            name: weights.grad_batch.clone().view(weights.grad_batch.size(0), -1)
            for name, weights in classifier.named_parameters()
        }

        dict_grad_statistics = {}
        for name, env_grads in dict_grads.items():
            assert "notcentered" not in flags.algorithm.split("_")
            env_mean = env_grads.mean(dim=0, keepdim=True)
            env_grads = env_grads - env_mean
            assert "offdiagonal" in flags.algorithm.split("_")
            # covariance considers components off-diagonal
            dict_grad_statistics[name] = torch.einsum("na,nb->ab", env_grads, env_grads
                                                     ) / (env_grads.size(0) * env_grads.size(1))

        return dict_grad_statistics

    def mmd(x, y, weights_x=None, weights_y=None):

        if flags.algorithm == "gaussian":
            raise ValueError("gaussian")
            # Kxx = gaussian_kernel(x, x).mean()
            # Kyy = gaussian_kernel(y, y).mean()
            # Kxy = gaussian_kernel(x, y).mean()
            # return Kxx + Kyy - 2 * Kxy

        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)

        transf_x = x
        transf_y = y
        if "notcentered" not in flags.algorithm.split("_"):
            transf_x = transf_x - mean_x
            transf_y = transf_y - mean_y

        if "weight" in flags.algorithm.split("_"):
            assert weights_x is not None
            transf_x = transf_x * weights_x.view((weights_x.size(0), 1))
            transf_y = transf_y * weights_y.view((weights_y.size(0), 1))

        if "offdiagonal" in flags.algorithm.split("_"):
            # cova_x = (transf_x.t() @ transf_x) / (transf_x.size(0) * transf_x.size(1))
            # cova_y = (transf_y.t() @ transf_y) / (transf_y.size(0) * transf_y.size(1))
            cova_x = torch.einsum("na,nb->ab", transf_x,
                                  transf_x) / (transf_x.size(0) * transf_x.size(1))
            cova_y = torch.einsum("na,nb->ab", transf_y,
                                  transf_y) / (transf_y.size(0) * transf_y.size(1))
        else:
            cova_x = (transf_x).pow(2).mean(dim=0)
            cova_y = (transf_y).pow(2).mean(dim=0)

        mean_diff = (mean_x - mean_y).pow(2).sum()#.mean()
        cova_diff = (cova_x - cova_y).pow(2).sum()#.mean()

        if "mean" in flags.algorithm.split("_"):
            return cova_diff + mean_diff
        return cova_diff

    def l2_between_dicts(dict_1, dict_2):
        assert len(dict_1) == len(dict_2)
        dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
        dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
        return (
            torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
            torch.cat(tuple([t.view(-1) for t in dict_2_values]))
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
            env['nll'] = compute_nll(logits, env['labels']).mean()
            env['acc'] = mean_accuracy(logits, env['labels'])

            if edx in [0, 1]:
                # when the dataset is in training
                optimizer.zero_grad()

                if "classcond" in flags.algorithm.split("_"):
                    env["grad_variance"] = {}
                    env['irm'] = {}
                    env['nllrex'] = {}
                    env['losses'] = {}
                    env['features'] = {}

                    for label in [0, 1]:
                        idx = (env['labels'] == label).view(-1)
                        env['losses'][label] = compute_mse(logits[idx], env['labels'][idx])

                        env['nllrex'][label] = compute_nll(logits[idx], env['labels'][idx]).mean()
                        env['irm'][label] = compute_irm_penalty(logits[idx], env['labels'][idx])
                        env["features"][label] = features[label]
                        if (
                            "alllayers" in flags.algorithm.split("_") or
                            "onlyextractor" in flags.algorithm.split("_")
                        ):
                            env["grad_statistics"] = compute_grad_variance(
                                mlp.prepare_input(env["images"][idx]), env['labels'][idx],
                                mlp.alllayers
                            )
                        else:
                            if "offdiagonal" in flags.algorithm.split("_"):
                                env["grad_statistics"] = compute_grad_covariance(
                                    features[idx], env['labels'][idx], mlp.classifier
                                )
                            else:
                                env["grad_statistics"] = compute_grad_variance(
                                    features[idx], env['labels'][idx], mlp.classifier
                                )
                else:
                    env['irm'] = compute_irm_penalty(logits, env['labels'])
                    env["features"] = features
                    env['losses'] = compute_mse(logits, env['labels'])
                    if (
                        "alllayers" in flags.algorithm.split("_") or
                        "onlyextractor" in flags.algorithm.split("_")
                    ):
                        env["grad_statistics"] = compute_grad_variance(
                            mlp.prepare_input(env["images"]), env['labels'], mlp.alllayers
                        )
                    else:
                        # assert "onlyclassifier" in flags.algorithm.split("_")
                        if "offdiagonal" in flags.algorithm.split("_"):
                            env["grad_statistics"] = compute_grad_covariance(
                                features, env['labels'], mlp.classifier
                            )
                        else:
                            env["grad_statistics"] = compute_grad_variance(
                                features, env['labels'], mlp.classifier
                            )

        train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
        train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()

        weight_norm = torch.tensor(0.).cuda()
        for w in mlp.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_nll.clone()
        loss += flags.l2_regularizer_weight * weight_norm

        if "classcond" in flags.algorithm.split("_"):
            fishr_penalty = 0
            irm_penalty = 0
            rex_penalty = 0
            coral_penalty = 0
            for label in [0, 1]:
                irm_penalty += torch.stack([envs[0]['irm'][label], envs[1]['irm'][label]]).mean()
                rex_penalty += (
                    envs[0]['nllrex'][label].mean() - envs[1]['nllrex'][label].mean()
                )**2
                coral_penalty += mmd(
                    envs[0]['features'][label], envs[1]['features'][label],
                    envs[0]['losses'][label], envs[1]['losses'][label]
                )

                if "fishr" in flags.algorithm.split("_"):
                    dict_grad_statistics_averaged = {
                        name: torch.stack(
                            [
                                envs[0]["grad_statistics"][label][name],
                                envs[1]["grad_statistics"][label][name]
                            ],
                            dim=0
                        ).mean(dim=0) for name in envs[0]["grad_statistics"][0]
                    }
                    fishr_penalty += (
                        l2_between_dicts(
                            envs[0]["grad_statistics"][label], dict_grad_statistics_averaged
                        ) + l2_between_dicts(
                            envs[1]["grad_statistics"][label], dict_grad_statistics_averaged
                        )
                    )
                else:
                    fishr_penalty = torch.tensor(0)

        else:
            irm_penalty = torch.stack([envs[0]['irm'], envs[1]['irm']]).mean()
            rex_penalty = (envs[0]['nll'].mean() - envs[1]['nll'].mean())**2

            coral_penalty = mmd(
                envs[0]['features'], envs[1]['features'], envs[0]['losses'], envs[1]['losses']
            )
            # Compute the gradient variance averaged over the two training domains
            dict_grad_statistics_averaged = {
                name: torch.stack(
                    [envs[0]["grad_statistics"][name], envs[1]["grad_statistics"][name]], dim=0
                ).mean(dim=0) for name in envs[0]["grad_statistics"]
            }
            fishr_penalty = (
                l2_between_dicts(envs[0]["grad_statistics"], dict_grad_statistics_averaged) +
                l2_between_dicts(envs[1]["grad_statistics"], dict_grad_statistics_averaged)
            )

        if flags.algorithm == "erm":
            pass
        else:
            # apply the selected regularization
            if flags.algorithm.startswith("fishr"):
                train_penalty = fishr_penalty
            elif flags.algorithm.startswith("coral"):
                train_penalty = coral_penalty
            elif flags.algorithm.startswith("rex"):
                train_penalty = rex_penalty
            elif flags.algorithm.startswith("irm"):
                train_penalty = irm_penalty
            else:
                raise ValueError(flags.algorithm)
            penalty_weight = (flags.penalty_weight if step >= flags.penalty_anneal_iters else 1.0)
            if step > 500 and "linearincrease" in flags.algorithm.split("_"):
                penalty_weight = penalty_weight * (step / 500)
            loss += penalty_weight * train_penalty
            if penalty_weight > 1.0:
                # Rescale the entire loss to keep backpropagated gradients in a reasonable range
                loss /= penalty_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        test_acc = envs[2]['acc']
        grayscale_test_acc = envs[3]['acc']
        if step % 100 == 0:
            pretty_print(
                np.int32(step),
                train_nll.detach().cpu().numpy(),
                train_acc.detach().cpu().numpy(),
                fishr_penalty.detach().cpu().numpy(),
                rex_penalty.detach().cpu().numpy(),
                irm_penalty.detach().cpu().numpy(),
                coral_penalty.detach().cpu().numpy(),
                test_acc.detach().cpu().numpy(),
                grayscale_test_acc.detach().cpu().numpy(),
            )

    final_train_accs.append(train_acc.detach().cpu().numpy())
    final_test_accs.append(test_acc.detach().cpu().numpy())
    final_graytest_accs.append(grayscale_test_acc.detach().cpu().numpy())
    print('Final train acc (mean/std across restarts so far):')
    print(np.mean(final_train_accs), np.std(final_train_accs))
    print('Final test acc (mean/std across restarts so far):')
    print(np.mean(final_test_accs), np.std(final_test_accs))
    print('Final gray test acc (mean/std across restarts so far):')
    print(np.mean(final_graytest_accs), np.std(final_graytest_accs))
