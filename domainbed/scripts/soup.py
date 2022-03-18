# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed 2 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/swaensshhpdeoa0316
# PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --test_envs 0 --trial_seed 2 --output_dir /data/rame/experiments/domainbed/erm66shhpeoa0317/ --mode ens

# Env variables to be considered
# CUDA_VISIBLE_DEVICES
# PRETRAINED
# NETMEMBER
# SWAMEMBER

import argparse
import itertools
import os
import json
import random
import numpy as np
import torch
import torch.utils.data
from domainbed import datasets, algorithms_inference
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader


def create_splits(inf_args, dataset):
    splits = []
    names = []
    inf_env = inf_args.inf_env.split("_")
    for env_i, env in enumerate(dataset):
        doit = "test" in inf_env and env_i in inf_args.test_envs
        doit = doit or ("train" in inf_env and env_i not in inf_args.test_envs)
        doit = doit or (str(env_i) in inf_env and env_i not in inf_args.test_envs)
        if not doit:
            continue

        out_, in_ = misc.split_dataset(
            env, int(len(env) * inf_args.holdout_fraction),
            misc.seed_hash(inf_args.trial_seed, env_i)
        )
        if "in" in inf_env:
            splits.append(in_)
            names.append('env{}_in'.format(env_i))

        if "out" in inf_env:
            splits.append(out_)
            names.append('env{}_out'.format(env_i))
    return splits, names

def get_testiid_score(results, keyacc, test_envs):
    if not results:
        return 0.
    results = json.loads(results)
    val_env_keys = []
    for i in itertools.count():
        if not keyacc:
            acc_key = f'env{i}_out_acc'
        else:
            acc_key = f'env{i}_out_Accuracies/acc_{keyacc}'
        if acc_key in results:
            if i not in test_envs:
                val_env_keys.append(acc_key)
        else:
            break
    return np.mean([results[key] for key in val_env_keys])


class NameSpace(object):

    def __init__(self, adict):
        self.__dict__.update(adict)

def _get_args():
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--algorithm', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--test_envs', type=int, nargs='+')
    parser.add_argument('--inf_env', type=str, default="test_in")
    parser.add_argument(
        '--trial_seed',
        type=int,
        default=-1,
        help='Trial number (used for seeding split_dataset and random_hparams).'
    )
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--data_dir', type=str, default="default")
    parser.add_argument('--mode', type=str, default="1by1")

    # select which folders
    parser.add_argument('--keyacc', type=str, default=None)
    parser.add_argument('--topk', type=int, default=0)

    return parser.parse_args()


def main():
    inf_args = _get_args()
    print(f"Begin soup for {inf_args}")
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
    splits, names = create_splits(inf_args, dataset)

    # load args
    folders = [
        os.path.join(output_dir, path) for output_dir in inf_args.output_dir.split(",") for path in os.listdir(output_dir)
    ]
    good_folders = {}
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        name_folder = os.path.split(folder)[-1]
        model_path = os.path.join(folder, "model.pkl")
        if not os.path.exists(model_path):
            print(f"absent: {name_folder}")
            continue
        save_dict = torch.load(model_path)
        train_args = NameSpace(save_dict["args"])

        if (
            train_args.dataset != inf_args.dataset or
            train_args.test_envs != inf_args.test_envs or
            (train_args.trial_seed != inf_args.trial_seed and inf_args.trial_seed != -1) or
            train_args.holdout_fraction != inf_args.holdout_fraction
        ):
            print(f"bad: {name_folder}")
            continue

        print(f"good: {name_folder}")
        # TODO
        proxy_perf = get_testiid_score(
            save_dict.get("results", ""),
            keyacc=inf_args.keyacc,
            test_envs=inf_args.test_envs)
        good_folders[folder] = proxy_perf

    if len(good_folders) == 0:
        return

    if inf_args.topk != 0:
        print(f"Select {inf_args.topk} checkpoints out of {len(good_folders)}")
        good_folders = sorted(good_folders.keys(), key=lambda x: good_folders[x], reverse=True)[:inf_args.topk]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if inf_args.mode == "1by1":
        for folder in good_folders:
            print(f"Inference at folder: {folder}")
            save_dict = torch.load(os.path.join(folder, "model.pkl"))
            train_args = NameSpace(save_dict["args"])
            random.seed(train_args.seed)
            np.random.seed(train_args.seed)
            torch.manual_seed(train_args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # load model
            algorithm_class = algorithms_inference.get_algorithm_class(train_args.algorithm)
            algorithm = algorithm_class(
                dataset.input_shape, dataset.num_classes,
                len(dataset) - len(inf_args.test_envs),
                hparams=save_dict["model_hparams"]
            )
            algorithm._init_from_save_dict(save_dict)
            algorithm.to(device)

            eval_loaders = [
                FastDataLoader(dataset=split, batch_size=64, num_workers=dataset.N_WORKERS)
                for split in splits
            ]

            results = {}
            evals = zip(names, eval_loaders)
            for i, (name, loader) in enumerate(evals):
                print(f"Inference at {name}")
                acc = algorithm.accuracy(
                    loader,
                    device,
                    compute_trace=False,
                    update_temperature=False,
                    output_temperature=(i == len(names) - 1)
                )
                for key in acc:
                    results[name + f'_{key}'] = acc[key]

            results_keys = sorted(results.keys())
            printed_keys = [key for key in results_keys if "diversity" not in key.lower()]
            misc.print_row([key.split("/")[-1] for key in printed_keys], colwidth=12, latex=True)
            misc.print_row([results[key] for key in printed_keys], colwidth=12, latex=True)

    elif inf_args.mode == "ens":
        ens_algorithm_class = algorithms_inference.get_algorithm_class(inf_args.algorithm)
        ens_algorithm = ens_algorithm_class(
            dataset.input_shape, dataset.num_classes,
            len(dataset) - len(inf_args.test_envs),
        )
        for folder in good_folders:
            print(f"Ingredient from folder: {os.path.split(folder)[-1]}")
            save_dict = torch.load(os.path.join(folder, "model.pkl"))
            train_args = NameSpace(save_dict["args"])

            # load model
            algorithm_class = algorithms_inference.get_algorithm_class(train_args.algorithm)
            algorithm = algorithm_class(
                dataset.input_shape, dataset.num_classes,
                len(dataset) - len(inf_args.test_envs),
                hparams=save_dict["model_hparams"]
            )
            algorithm._init_from_save_dict(save_dict)
            ens_algorithm.add_new_algorithm(algorithm)

            del algorithm

        ens_algorithm.to(device)
        random.seed(train_args.seed)
        np.random.seed(train_args.seed)
        torch.manual_seed(train_args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        eval_loaders = [
            FastDataLoader(dataset=split, batch_size=64, num_workers=dataset.N_WORKERS)
            for split in splits
        ]
        results = {}
        evals = zip(names, eval_loaders)
        for i, (name, loader) in enumerate(evals):
            print(f"Inference at {name}")
            acc = ens_algorithm.accuracy(
                loader,
                device,
                compute_trace=False,
                update_temperature=False,
                output_temperature=(i == len(names) - 1)
            )
            for key in acc:
                results[name + f'_{key}'] = acc[key]

        results_keys = sorted(results.keys())
        print(f"Results for {inf_args} with {len(good_folders)}")
        misc.print_row([key.split("/")[-1] for key in results_keys], colwidth=12, latex=True)
        misc.print_row([results[key] for key in results_keys], colwidth=12, latex=True)
    else:
        raise ValueError(inf_args.mode)

if __name__ == "__main__":
    main()
