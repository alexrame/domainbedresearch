# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Ensembling --dataset OfficeHome --test_envs 0 --trial_seed 2 --output_dir /data/rame/experiments/domainbed/swaensshhpdeoa0316/0583a640ee2afc0cd74c88540ba06bad/

import argparse
import os
import random
import numpy as np
import torch
import torch.utils.data
from domainbed import datasets, algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader

class NameSpace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

def main():
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--algorithm', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--test_envs', type=int, nargs='+')
    parser.add_argument(
        '--trial_seed',
        type=int,
        help='Trial number (used for seeding split_dataset and random_hparams).'
    )
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--data_dir', type=str, default="default")
    inf_args = parser.parse_args()

    # load model
    assert os.path.exists(inf_args.output_dir)
    save_dict = torch.load(os.path.join(inf_args.output_dir, "model.pkl"))
    train_args = NameSpace(save_dict["args"])

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

        in_splits.append((in_))
        out_splits.append((out))
        if len(uda):
            uda_splits.append((uda))

    if train_args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    print("Train Envs:", [env for env in enumerate(in_splits) if env not in inf_args.test_envs])

    eval_loaders = [
        FastDataLoader(dataset=env, batch_size=64, num_workers=dataset.N_WORKERS)
        for env in (in_splits + out_splits + uda_splits)
    ]
    eval_loader_names = ['env{}_in'.format(i) for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i) for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i) for i in range(len(uda_splits))]

    results = {}

    evals = zip(eval_loader_names, eval_loaders)
    for name, loader in evals:
        acc = algorithm.accuracy(
            loader, device, compute_trace=False, update_temperature=False
        )
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
