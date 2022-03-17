# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import os
import random
import numpy as np
import torch
import torch.utils.data
from domainbed import datasets, algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader

def main():
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--data_dir', type=str, default="default")
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--algorithm', type=str)
    parser.add_argument('--test_envs', type=int, nargs='+')
    parser.add_argument(
        '--trial_seed',
        type=int,
        help='Trial number (used for seeding split_dataset and random_hparams).'
    )
    inf_args = parser.parse_args()

    # load model
    assert os.path.exists(inf_args.output_dir)
    save_dict = torch.load(os.path.join(inf_args.output_dir, "model.pkl"))
    train_args = save_dict["args"]

    assert train_args.dataset == inf_args.dataset
    assert train_args.algorithm == inf_args.algorithm
    assert train_args.test_envs == inf_args.test_envs
    assert train_args.trial_seed == inf_args.trial_seed

    hparams = save_dict["model_hparams"]

    if inf_args.data_dir == "default":
        if "DATA" in os.environ:
            inf_args.data_dir = os.path.join(os.environ["DATA"], "data/domainbed/")
        else:
            inf_args.data_dir = "domainbed/data"

    random.seed(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if inf_args.dataset in vars(datasets):
        dataset_class = vars(datasets)[inf_args.dataset]
        dataset = dataset_class(inf_args.data_dir, inf_args.test_envs, hparams=hparams)
    else:
        raise NotImplementedError

    algorithm_class = algorithms.get_algorithm_class(inf_args.algorithm)
    algorithm = algorithm_class(
        dataset.input_shape, dataset.num_classes,
        len(dataset) - len(inf_args.test_envs), hparams
    )

    algorithm._init_from_save_dict(save_dict)
    algorithm.to(device)

    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        if not algorithm_class.CUSTOM_FORWARD and dataset_class.CUSTOM_DATASET:
            env = misc.CustomToRegularDataset(env)

        out, in_ = misc.split_dataset(
            env, int(len(env) * train_args.holdout_fraction), misc.seed_hash(train_args.trial_seed, env_i)
        )

        if env_i in train_args.test_envs:
            uda, in_ = misc.split_dataset(
                in_, int(len(in_) * train_args.uda_holdout_fraction),
                misc.seed_hash(train_args.trial_seed, env_i)
            )
            # uda, out = misc.split_dataset(out, int(len(out)*args.uda_holdout_fraction),
            #                               misc.seed_hash(args.trial_seed, env_i))

        in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if train_args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    print("Train Envs:", [i for (i, _) in enumerate(in_splits) if i not in inf_args.test_envs])

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
                traced_envs = [inf_args.test_envs[0], inf_args.test_envs[0] +
                                1] if inf_args.test_envs[0] != 3 else [1, 3]
                compute_trace = any([("env" + str(env)) in name for env in traced_envs])
                # ((step % (6 * checkpoint_freq) == 0) or (step == n_steps - 1))
            else:
                compute_trace = False
            update_temperature = name in [
                'env{}_out'.format(i)
                for i in range(len(out_splits))
                if i not in inf_args.test_envs
            ]
            acc = algorithm.accuracy(
                loader, device, compute_trace, update_temperature=update_temperature
            )
        else:
            acc = misc.accuracy(algorithm, loader, weights, device)
        for key in acc:
            results[name + f'_{key}'] = acc[key]

    results_keys = sorted(results.keys())
    printed_keys = [key for key in results_keys if "Diversity" not in key.lower()]
    if results_keys != last_results_keys:
        misc.print_row([key.split("/")[-1] for key in printed_keys], colwidth=12, latex=True)
        last_results_keys = results_keys
    misc.print_row([results[key] for key in printed_keys], colwidth=12, latex=True)




if __name__ == "__main__":
    main()
