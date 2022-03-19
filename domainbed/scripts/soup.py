# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed 2 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/swaensshhpdeoa0316
# PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --test_envs 0 --trial_seed 2 --output_dir /data/rame/experiments/domainbed/erm66shhpeoa0317/ --mode ens

# Env variables to be considered
# CUDA_VISIBLE_DEVICES
# PRETRAINED
# NETMEMBER
# SWAMEMBER
# HESSIAN
# HESSIANBS

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
    found_checkpoints_per_cluster = find_checkpoints(inf_args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    good_checkpoints = []
    for cluster, found_checkpoints in found_checkpoints_per_cluster.items():
        print(f"Exploring cluster: {cluster} with {len(found_checkpoints)} checkpoints")
        if inf_args.mode == "greedy":
            if "trial_seed" in inf_args.cluster:
                assert inf_args.selection == "train"
                trial_seed = int(cluster.split("|")[inf_args.cluster.index("trial_seed")])
            else:
                trial_seed = inf_args.trial_seed
            val_splits, val_names = create_splits(
                inf_args,
                dataset,
                inf_env="train" if inf_args.selection == "train" else "test",
                filter="out",
                trial_seed=trial_seed
            )
            cluster_good_checkpoints = get_greedy_checkpoints(
                found_checkpoints, dataset, inf_args, val_names, val_splits, device
            )
        elif inf_args.topk != 0:
            cluster_good_checkpoints = found_checkpoints[:inf_args.topk]
        else:
            cluster_good_checkpoints = found_checkpoints[:]
        print(f"Select {len(cluster_good_checkpoints)}/{len(found_checkpoints)} checkpoints")
        good_checkpoints.extend(cluster_good_checkpoints)

    ood_splits, ood_names = create_splits(
        inf_args,
        dataset,
        inf_env="test",
        filter="full" if inf_args.selection == "train" else "in",
        trial_seed=inf_args.trial_seed
    )
    get_results_for_checkpoints(good_checkpoints, dataset, inf_args, ood_names, ood_splits, device)


def _get_args():
    parser = argparse.ArgumentParser(description='Domain generalization')

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

    # select which checkpoints
    parser.add_argument('--mode', type=str, default="ens")  # or "greedy"
    parser.add_argument(
        '--cluster',
        type=str,
        default=[],
        nargs='+',
    ) # algorithm trial_seed
    parser.add_argument(
        '--regexes',
        type=str,
        default=[],
        nargs='+',
    ) # net0_net1

    parser.add_argument('--criteriontopk', type=str, default="acc_net")
    parser.add_argument('--topk', type=int, default=0)
    parser.add_argument('--selection', type=str, default="train")  # or "oracle"

    parser.add_argument('--algorithm', type=str)
    parser.add_argument('--t_scaled', type=str)

    inf_args = parser.parse_args()
    if inf_args.data_dir == "default":
        if "DATA" in os.environ:
            inf_args.data_dir = os.path.join(os.environ["DATA"], "data/domainbed/")
        else:
            inf_args.data_dir = "domainbed/data"

    return inf_args


def create_splits(inf_args, dataset, inf_env, filter, trial_seed):
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
                env, int(len(env) * inf_args.holdout_fraction), misc.seed_hash(trial_seed, env_i)
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


def get_score_run(results, criteriontopk, test_envs):
    if not results:
        return 0.
    if criteriontopk in ["none", "0"]:
        return 0.

    if criteriontopk.startswith("acc"):
        criteriontopk = "Accuracies/{criteriontopk}"
    elif criteriontopk.startswith("ece"):
        criteriontopk = "Calibration/{criteriontopk}"
    else:
        raise ValueError(f"Unknown criterion {criteriontopk}")

    results = json.loads(results)
    val_env_keys = []
    for i in itertools.count():
        acc_key = f'env{i}_out_{criteriontopk}'
        if acc_key in results:
            if i not in test_envs:
                val_env_keys.append(acc_key)
        else:
            break
    return np.mean([results[key] for key in val_env_keys])


def find_checkpoints(inf_args, verbose=False):
    checkpoints = [
        os.path.join(output_dir, path)
        for output_dir in inf_args.output_dir.split(",")
        for path in os.listdir(output_dir)
    ]
    found_checkpoints_per_cluster = {}
    for folder in checkpoints:
        if not os.path.isdir(folder):
            continue
        name_folder = os.path.split(folder)[-1]
        model_path = os.path.join(folder, "model.pkl")
        if not os.path.exists(model_path):
            if verbose:
                print(f"absent: {name_folder}")
            continue
        save_dict = torch.load(model_path)
        train_args = NameSpace(save_dict["args"])

        if train_args.dataset != inf_args.dataset:
            if verbose:
                print(f"bad dataset: {name_folder}")
            continue
        if train_args.test_envs != inf_args.test_envs:
            if verbose:
                print(f"bad test env: {name_folder}")
            continue
        if (train_args.trial_seed != inf_args.trial_seed and inf_args.trial_seed != -1):
            if verbose:
                print(f"bad trial seed: {name_folder}")
            continue
        if train_args.holdout_fraction != inf_args.holdout_fraction:
            if verbose:
                print(f"Warning different holdout fraction: {name_folder} but keep")

        if verbose:
            print(f"found: {name_folder}")
        score_folder = get_score_run(
            save_dict.get("results", ""),
            criteriontopk=inf_args.criteriontopk,
            test_envs=inf_args.test_envs
        )
        cluster = "|".join([str(train_args.__dict__[cluster]) for cluster in inf_args.cluster])
        if cluster not in found_checkpoints_per_cluster:
            found_checkpoints_per_cluster[cluster] = {}
        found_checkpoints_per_cluster[cluster][folder] = score_folder

    if len(found_checkpoints_per_cluster) == 0:
        raise ValueError("No checkpoints found")
        return []
    print(found_checkpoints_per_cluster)

    found_checkpoints_per_cluster = {
        cluster: sorted(found_checkpoints.keys(), key=lambda x: found_checkpoints[x], reverse=True)
        for cluster, found_checkpoints in found_checkpoints_per_cluster.items()
    }
    return found_checkpoints_per_cluster


def file_with_weights(folder):
    filename = os.path.join(folder, "model.pkl")
    filename_heavy = os.path.join(folder, "model_with_weights.pkl")
    if os.path.exists(filename_heavy):
        filename = filename_heavy
    else:
        assert os.path.exists(filename)
    return filename

def get_greedy_checkpoints(found_checkpoints, dataset, inf_args, val_names, val_splits, device):

    ens_algorithm_class = algorithms_inference.get_algorithm_class(inf_args.algorithm)
    ens_algorithm = ens_algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(inf_args.test_envs),
    )
    best_results = {}
    good_nums = []
    for num, folder in enumerate(found_checkpoints):
        # print(f"Ingredient {num} from folder: {os.path.split(folder)[-1]}")

        save_dict = torch.load(file_with_weights(folder))
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
            # print(f"Inference at {name} at num {num}")
            results_of_one_eval = ens_algorithm.accuracy(
                loader,
                device,
                compute_trace=False,
                update_temperature=False,
                output_temperature=False
            )
            for key in results_of_one_eval:
                val_results[key] = val_results.get(key,
                                                   0) + results_of_one_eval[key] / len(val_names)

        # print(f"Val results for {inf_args} at {num}")
        results_keys = sorted(val_results.keys())
        # misc.print_row([key.split("/")[-1] for key in results_keys], colwidth=15, latex=True)
        # misc.print_row([val_results[key] for key in results_keys], colwidth=15, latex=True)

        for key in val_results:
            if not key.startswith("Accuracies"):
                continue
            if val_results[key] > best_results.get(key, ([], 0.))[1]:
                if key == f"Accuracies/acc_soup":
                    good_nums.append(num)
                best_results[key] = (good_nums[:], val_results[key])

        if num not in good_nums:
            ens_algorithm.delete_last()
            print(f"Skip num {num}")
        else:
            print(f"Add num {num}")
        # print("")

    print(f"Best OOD results for {inf_args} with {len(good_nums)} checkpoints")
    print(best_results)
    return [found_checkpoints[num] for num in good_nums]


def get_results_for_checkpoints(good_checkpoints, dataset, inf_args, ood_names, ood_splits, device):
    ens_algorithm_class = algorithms_inference.get_algorithm_class(inf_args.algorithm)
    ens_algorithm = ens_algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(inf_args.test_envs),
        t_scaled=inf_args.t_scaled,
        regexes=inf_args.regexes,
    )
    for folder in good_checkpoints:
        print(f"Ingredient from folder: {folder}")
        save_dict = torch.load(file_with_weights(folder))
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
    ens_algorithm.eval()
    random.seed(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ood_loaders = [
        FastDataLoader(
            dataset=split, batch_size=64, num_workers=dataset.N_WORKERS
        ) for split in ood_splits
    ]
    compute_hessian = os.environ.get("HESSIAN") != "0"
    if compute_hessian:
        fraction = float(os.environ.get("HESSIAN", 0.2))
        ood_splits_small = [
            misc.split_dataset(split, int(len(split) * fraction), 0)[0] for split in ood_splits
        ]
        ood_loaders_small = [
            FastDataLoader(
                dataset=split,
                batch_size=int(os.environ.get("HESSIANBS", 12)),
                num_workers=dataset.N_WORKERS
            ) for split in ood_splits_small
        ]
    evals = zip(ood_names, ood_loaders)
    ood_results = {}
    for i, (name, loader) in enumerate(evals):
        print(f"Inference at {name}")

        results = ens_algorithm.accuracy(
            loader, device, compute_trace=True)
        print(results)

        if compute_hessian:
            loader_small = ood_loaders_small[i]
            print(f"Begin Hessian for loaders of len: {len(loader_small)}")
            assert len(ood_names) == 1
            del ens_algorithm.swas[1:]
            del ens_algorithm.networks[1:]

            print(f"Begin Hessian soup")
            results["Flatness/souphess"] = misc.compute_hessian(
                ens_algorithm.soup.network_soup, loader_small, maxIter=10
            )
            del ens_algorithm.soup.network_soup

            # print(f"Begin Hessian soupswa")
            # results["Flatness/soupswahess"] = misc.compute_hessian(
            #     ens_algorithm.soupswa.network_soup, loader_small, maxIter=10
            # )
            del ens_algorithm.soupswa.network_soup

            print("Begin Hessian swa0")
            results[f"Flatness/swa0hess"] = misc.compute_hessian(
                ens_algorithm.swas[0], loader_small, maxIter=10
            )
            del ens_algorithm.swas[0]

            print("Begin Hessian net0")
            results[f"Flatness/net0hess"] = misc.compute_hessian(
                ens_algorithm.networks[0], loader_small, maxIter=10
            )
            del ens_algorithm.networks[0]

        for key in results:
            ood_results[name + "_" + key.split("/")[-1]] = results[key]

    print(f"OOD results for {inf_args} with {len(good_checkpoints)}")
    ood_results_keys = sorted(ood_results.keys())
    misc.print_row(ood_results_keys, colwidth=15, latex=True)
    misc.print_row([ood_results[key] for key in ood_results_keys], colwidth=15, latex=True)


if __name__ == "__main__":
    main()
