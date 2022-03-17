# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Ensembling --dataset OfficeHome --test_envs 0 --trial_seed 2 --output_dir /data/rame/experiments/domainbed/swaensshhpdeoa0316/0583a640ee2afc0cd74c88540ba06bad/

import argparse
import os
import random
import numpy as np
import torch
import torch.utils.data
from domainbed import datasets, algorithms_inference
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader


def create_splits(inf_args, dataset):
    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        if inf_args.inf_env == "test" and env_i not in inf_args.test_envs:
            continue
        if inf_args.inf_env == "train" and env_i in inf_args.test_envs:
            continue

        out, in_ = misc.split_dataset(
            env, int(len(env) * inf_args.holdout_fraction),
            misc.seed_hash(inf_args.trial_seed, env_i)
        )

        in_splits.append((in_))
        out_splits.append((out))
    return in_splits, out_splits


class NameSpace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

def main():
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--algorithm', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--test_envs', type=int, nargs='+')
    parser.add_argument('--inf_env', type=str, default="test")
    parser.add_argument(
        '--trial_seed',
        type=int,
        help='Trial number (used for seeding split_dataset and random_hparams).'
    )
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--data_dir', type=str, default="default")
    inf_args = parser.parse_args()



    if inf_args.data_dir == "default":
        if "DATA" in os.environ:
            inf_args.data_dir = os.path.join(os.environ["DATA"], "data/domainbed/")
        else:
            inf_args.data_dir = "domainbed/data"

    if inf_args.dataset in vars(datasets):
        dataset_class = vars(datasets)[inf_args.dataset]
        dataset = dataset_class(
            inf_args.data_dir, inf_args.test_envs, hparams={"data_augmentation": True}
        )
    else:
        raise NotImplementedError
    in_splits, out_splits = create_splits(inf_args, dataset)

    algorithm_class = algorithms_inference.get_algorithm_class(inf_args.algorithm)


    # load args
    train_args = NameSpace(save_dict["args"])

    assert train_args.dataset == inf_args.dataset
    assert train_args.algorithm == inf_args.algorithm
    assert train_args.test_envs == inf_args.test_envs
    assert train_args.trial_seed == inf_args.trial_seed
    assert train_args.holdout_fraction == inf_args.holdout_fraction

    random.seed(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    assert os.path.exists(inf_args.output_dir)

    # load model
    save_dict = torch.load(os.path.join(inf_args.output_dir, "model.pkl"))
    hparams = save_dict["model_hparams"]

    algorithm = algorithm_class(
        dataset.input_shape, dataset.num_classes,
        len(dataset) - len(inf_args.test_envs), hparams
    )
    algorithm._init_from_save_dict(save_dict)
    algorithm.to(device)

    eval_loaders = [
        FastDataLoader(dataset=env, batch_size=64, num_workers=dataset.N_WORKERS)
        for env in (in_splits + out_splits)
    ]
    eval_loader_names = ['env{}_in'.format(i) for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i) for i in range(len(out_splits))]
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
    misc.print_row([key.split("/")[-1] for key in printed_keys], colwidth=12, latex=True)
    misc.print_row([results[key] for key in printed_keys], colwidth=12, latex=True)


if __name__ == "__main__":
    main()
