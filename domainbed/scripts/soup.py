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



def main():
    inf_args = _get_args()
    print(f"Begin soup for {inf_args}")

    if inf_args.dataset in vars(datasets):
        dataset_class = vars(datasets)[inf_args.dataset]
        dataset = dataset_class(
            inf_args.data_dir, inf_args.test_envs, hparams={"data_augmentation": True}
        )
    else:
        raise NotImplementedError

    # load args
    found_folders = find_folders(inf_args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if inf_args.mode == "greedy":
        val_splits, val_names = create_splits(
            inf_args,
            dataset,
            inf_env="train" if inf_args.selection == "train" else "test",
            filter="out"
        )
        good_folders = get_greedy_folders(
            found_folders, dataset, inf_args, val_names, val_splits, device
        )
    elif inf_args.topk != 0:
        print(f"Select {inf_args.topk} checkpoints out of {len(good_folders)}")
        good_folders = found_folders[:inf_args.topk]
    else:
        good_folders = found_folders[:]

    ood_splits, ood_names = create_splits(
        inf_args, dataset, inf_env="test", filter="full" if inf_args.selection == "train" else "in"
    )
    get_results_for_folders(good_folders, dataset, inf_args, ood_names, ood_splits, device)


def _get_args():
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--algorithm', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--test_envs', type=int, nargs='+')
    parser.add_argument(
        '--trial_seed',
        type=int,
        default=-1,
        help='Trial number (used for seeding split_dataset and random_hparams).'
    )
    parser.add_argument('--holdout_fraction', type=float, default=0.05)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--data_dir', type=str, default="default")
    parser.add_argument('--mode', type=str, default="1by1")

    # select which folders
    parser.add_argument('--keyacc', type=str, default="net")
    parser.add_argument('--topk', type=int, default=0)
    parser.add_argument('--selection', type=str, default="train")  # or "oracle"

    inf_args = parser.parse_args()
    if inf_args.data_dir == "default":
        if "DATA" in os.environ:
            inf_args.data_dir = os.path.join(os.environ["DATA"], "data/domainbed/")
        else:
            inf_args.data_dir = "domainbed/data"

    return inf_args


def create_splits(inf_args, dataset, inf_env, filter):
    splits = []
    names = []
    for env_i, env in enumerate(dataset):
        if inf_env == "test" and env_i not in inf_args.test_envs:
            continue
        if inf_env == "train" and env_i in inf_args.test_envs:
            continue

        if filter == "full":
            splits.append(env)
            names.append('e{}'.format(env_i))
        else:
            out_, in_ = misc.split_dataset(
                env, int(len(env) * inf_args.holdout_fraction),
                misc.seed_hash(inf_args.trial_seed, env_i)
            )
            if filter == "in":
                splits.append(in_)
                names.append('e{}_in'.format(env_i))
            elif filter == "out":
                splits.append(out_)
                names.append('e{}_out'.format(env_i))
            else:
                raise ValueError(filter)

    return splits, names

class NameSpace(object):

    def __init__(self, adict):
        self.__dict__.update(adict)

def get_testiid_score(results, keyacc, test_envs):
    if not results:
        return 0.
    results = json.loads(results)
    val_env_keys = []
    for i in itertools.count():
        acc_key = f'env{i}_out_Accuracies/acc_{keyacc}'
        if acc_key in results:
            if i not in test_envs:
                val_env_keys.append(acc_key)
        else:
            break
    return np.mean([results[key] for key in val_env_keys])

def find_folders(inf_args):
    folders = [
        os.path.join(output_dir, path)
        for output_dir in inf_args.output_dir.split(",")
        for path in os.listdir(output_dir)
    ]
    found_folders = {}
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
            train_args.dataset != inf_args.dataset or train_args.test_envs != inf_args.test_envs or
            (train_args.trial_seed != inf_args.trial_seed and inf_args.trial_seed != -1) or
            train_args.holdout_fraction != inf_args.holdout_fraction
        ):
            # print(f"bad: {name_folder}")
            continue

        print(f"found: {name_folder}")
        proxy_perf = get_testiid_score(
            save_dict.get("results", ""), keyacc=inf_args.keyacc, test_envs=inf_args.test_envs
        )
        found_folders[folder] = proxy_perf

    if len(found_folders) == 0:
        return

    found_folders = sorted(found_folders.keys(), key=lambda x: found_folders[x], reverse=True)
    return found_folders


def get_greedy_folders(found_folders, dataset, inf_args, val_names, val_splits, device):

    ens_algorithm_class = algorithms_inference.get_algorithm_class(inf_args.algorithm)
    ens_algorithm = ens_algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(inf_args.test_envs),
    )
    best_results = {}
    good_nums = []
    for num, folder in enumerate(found_folders):
        print(f"Ingredient {num} from folder: {os.path.split(folder)[-1]}")
        save_dict = torch.load(os.path.join(folder, "model.pkl"))
        train_args = NameSpace(save_dict["args"])

        # load model
        algorithm_class = algorithms_inference.get_algorithm_class(train_args.algorithm)
        algorithm = algorithm_class(
            dataset.input_shape,
            dataset.num_classes,
            len(dataset) - len(inf_args.test_envs),
            hparams=save_dict["model_hparams"]
        )
        algorithm._init_from_save_dict(save_dict)
        ens_algorithm.to("cpu")
        ens_algorithm.add_new_algorithm(algorithm)
        del algorithm
        ens_algorithm.to(device)

        random.seed(train_args.seed)
        np.random.seed(train_args.seed)
        torch.manual_seed(train_args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        val_loaders = [
            FastDataLoader(dataset=split, batch_size=64, num_workers=dataset.N_WORKERS)
            for split in val_splits
        ]
        val_results = {}
        val_evals = zip(val_names, val_loaders)
        for name, loader in val_evals:
            print(f"Inference at {name} at num {num}")
            results_of_one_eval = ens_algorithm.accuracy(
                loader,
                device,
                compute_trace=False,
                update_temperature=False,
                output_temperature=False
            )
            for key in results_of_one_eval:
                val_results[key] = val_results.get(key, 0) + results_of_one_eval[key] / len(val_names)

        print(f"Val results for {inf_args} at {num}")
        results_keys = sorted(val_results.keys())
        misc.print_row(results_keys, colwidth=15, latex=True)
        misc.print_row([val_results[key] for key in results_keys], colwidth=15, latex=True)

        for key in val_results:
            if val_results[key] > best_results.get(key, ([], 0.))[1]:
                if key == f"Accuracies/acc_{inf_args.keyacc}":
                    good_nums.append(num)
                best_results[key] = (good_nums[:], val_results[key])

        if num not in good_nums:
            ens_algorithm.delete_last()
            print(f"Skip num {num}")
        else:
            print(f"Add num {num}")

    print(f"Best OOD results for {inf_args} with {len(good_nums)} folders")
    print(best_results)
    return [found_folders[num] for num in good_nums]



def get_results_for_folders(good_folders, dataset, inf_args, ood_names, ood_splits, device):
    ens_algorithm_class = algorithms_inference.get_algorithm_class(inf_args.algorithm)
    ens_algorithm = ens_algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(inf_args.test_envs),
    )
    for folder in good_folders:
        print(f"Ingredient from folder: {folder}")
        save_dict = torch.load(os.path.join(folder, "model.pkl"))
        train_args = NameSpace(save_dict["args"])

        # load model
        hparams = save_dict["model_hparams"]
        algorithm_class = algorithms_inference.get_algorithm_class(train_args.algorithm)
        algorithm = algorithm_class(
            dataset.input_shape, dataset.num_classes,
            len(dataset) - len(inf_args.test_envs), hparams
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

    ood_loaders = [
        FastDataLoader(dataset=split, batch_size=64, num_workers=dataset.N_WORKERS)
        for split in ood_splits
    ]
    evals = zip(ood_names, ood_loaders)
    results = {}
    for i, (name, loader) in enumerate(evals):
        print(f"Inference at {name}")
        acc = ens_algorithm.accuracy(
            loader,
            device,
            compute_trace=False,
            update_temperature=False,
            output_temperature=(i == len(ood_names) - 1)
        )
        for key in acc:
            results[name + "_" + key.split("/")[-1]] = acc[key]


    print(f"OOD results for {inf_args} with {len(good_folders)}")
    results_keys = sorted(results.keys())
    misc.print_row(results_keys, colwidth=15, latex=True)
    misc.print_row([results[key] for key in results_keys], colwidth=15, latex=True)



if __name__ == "__main__":
    main()
