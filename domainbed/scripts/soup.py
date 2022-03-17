# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import statistics
import yaml
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
from domainbed import datasets, hparams_registry, algorithms
from domainbed.algorithms import Algorithm
from domainbed.lib import misc, experiments_handler
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader


def main():
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default="default")
    parser.add_argument('--dataset', type=str, default="ColoredMNIST")
    parser.add_argument('--algorithm', type=str, default="Ensembling")
    parser.add_argument('--output_dir', type=str)
    parser.add_argument(
        '--trial_seed',
        type=int,
        default=0,
        help='Trial number (used for seeding split_dataset and random_hparams).'
    )
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--test_envs', type=int, nargs='+')
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument(
        '--uda_holdout_fraction',
        type=float,
        default=0,
        help="For domain adaptation, % of test to use unlabeled for training."
    )
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')

    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.data_dir == "default":
        if "DATA" in os.environ:
            args.data_dir = os.path.join(os.environ["DATA"], "data/domainbed/")
        else:
            args.data_dir = "domainbed/data"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)

    if args.dataset in vars(datasets):
        dataset_class = vars(datasets)[args.dataset]
        if args.test_envs is None:
            args.test_envs = dataset_class.TEST_ENVS
        dataset = dataset_class(args.data_dir, args.test_envs, hparams=hparams)
    else:
        raise NotImplementedError

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(
        dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams
    )

    # load model
    assert os.path.exists(args.output_dir)
    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        if not algorithm_class.CUSTOM_FORWARD and dataset_class.CUSTOM_DATASET:
            env = misc.CustomToRegularDataset(env)

        out, in_ = misc.split_dataset(
            env, int(len(env) * args.holdout_fraction), misc.seed_hash(args.trial_seed, env_i)
        )

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(
                in_, int(len(in_) * args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i)
            )
            # uda, out = misc.split_dataset(out, int(len(out)*args.uda_holdout_fraction),
            #                               misc.seed_hash(args.trial_seed, env_i))

        in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    print("Train Envs:", [i for (i, _) in enumerate(in_splits) if i not in args.test_envs])

    eval_loaders = [
        FastDataLoader(dataset=env, batch_size=64, num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)
    ]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i) for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i) for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i) for i in range(len(uda_splits))]


    results = {}

    evals = zip(eval_loader_names, eval_loaders, eval_weights)
    for name, loader, weights in evals:
        if hasattr(algorithm, "accuracy"):
            if os.environ.get("HESSIAN") == "1":
                traced_envs = [args.test_envs[0], args.test_envs[0] +
                                1] if args.test_envs[0] != 3 else [1, 3]
                compute_trace = any([("env" + str(env)) in name for env in traced_envs])
                # ((step % (6 * checkpoint_freq) == 0) or (step == n_steps - 1))
            else:
                compute_trace = False
            update_temperature = name in [
                'env{}_out'.format(i)
                for i in range(len(out_splits))
                if i not in args.test_envs
            ]
            acc = algorithm.accuracy(
                loader, device, compute_trace, update_temperature=update_temperature
            )
        else:
            acc = misc.accuracy(algorithm, loader, weights, device)
        for key in acc:
            results[name + f'_{key}'] = acc[key]
            if "/" not in key:
                tb_name = f'{name}_{key}'
            else:
                tb_name = f'{key.split("/")[0]}/{name}_{key.split("/")[1]}'

    results_keys = sorted(results.keys())
    printed_keys = [key for key in results_keys if "Diversity" not in key.lower()]
    if results_keys != last_results_keys:
        misc.print_row([key.split("/")[-1] for key in printed_keys], colwidth=12)
        last_results_keys = results_keys
    misc.print_row([results[key] for key in printed_keys], colwidth=12)

    results.update({'hparams': hparams, 'args': vars(args)})





if __name__ == "__main__":
    main()
