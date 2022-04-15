# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Env variables to be considered
# Debugging: SWAMEMBER=0 NUMSTEPSTEMP=200 PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /data/rame/experiments/domainbed/erm24sheoa0319/2f1c184db1532874464b694a1142dd52/ --topk 30 --mode iter_1_12 --do_ens 1 --t_scaled temp_train --ood_data train,test
# CUDA_VISIBLE_DEVICES
# PRETRAINED
# NETMEMBER
# SWAMEMBER
# HESSIAN
# HESSIANFRAC
# HESSIANBS
# NUMSTEPSTEMP

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
from domainbed.lib import misc, experiments_handler

random.seed(os.environ.get('SEED', 4))

def gpuprint(*args, **kwargs):
    print(os.environ.get('CUDA_VISIBLE_DEVICES', "-1") + ":", *args, **kwargs)


def main():
    inf_args = _get_args()
    gpuprint(f"Begin soup for {inf_args}")

    if inf_args.dataset in vars(datasets):
        dataset_class = vars(datasets)[inf_args.dataset]
        dataset = dataset_class(
            inf_args.data_dir, inf_args.test_envs, hparams={"data_augmentation": True}
        )
    else:
        raise NotImplementedError

    # load args
    sorted_checkpoints_per_cluster, dict_checkpoints, dict_checkpoints_to_score = find_checkpoints(
        inf_args, verbose=os.environ.get("DEBUG", "0") != "0"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    good_checkpoints = get_good_checkpoints(
        sorted_checkpoints_per_cluster, inf_args, dataset, device, dict_checkpoints_to_score
    )
    if os.environ.get("DEBUG"):
        gpuprint("good_checkpoints", good_checkpoints)
        gpuprint("dict_checkpoints_to_score", dict_checkpoints_to_score)

    ood_splits, ood_names = [], []
    for ood_env in inf_args.ood_data.split(","):
        dict_env_to_filter = {}
        if ood_env == "test":
            # if inf_args.trial_seed == [-1]:
            dict_env_to_filter["test"] = "full" if inf_args.selection_data == "train" else "in"
            # else:
            #     dict_env_to_filter["test"] = "in"
        elif ood_env == "testout":
            # if inf_args.trial_seed == [-1]:
            dict_env_to_filter["test"] = "out"
        elif ood_env == "train":
            dict_env_to_filter["train"] = "out"
        elif ood_env == "trainf":
            dict_env_to_filter["train"] = "full"
        else:
            raise ValueError(ood_env)

        _ood_splits, _ood_names = create_splits(
            inf_args, dataset, dict_env_to_filter=dict_env_to_filter, trial_seed=inf_args.trial_seed[0]
        )
        if ood_env == "train":
            ood_splits.append(misc.MergeDataset(_ood_splits))
            ood_names.append("train_out")
        elif ood_env == "trainf":
            ood_splits.append(misc.MergeDataset(_ood_splits))
            ood_names.append("train_full")
        else:
            ood_splits.extend(_ood_splits)
            ood_names.extend(_ood_names)

    if os.environ.get("HESSIAN", "-1") != "-1":
        hessian_splits, hessian_names = [], []
        for hessian in os.environ.get("HESSIAN", "-1").split("_"):
            hessian_filter = os.environ.get("HESSIANFILTER", "in")
            _hessian_splits, _ = create_splits(
                inf_args,
                dataset,
                dict_env_to_filter={inf_env: hessian_filter for inf_env in hessian.split(",")},
                trial_seed=inf_args.trial_seed[0],
                holdout_fraction=float(os.environ.get("HESSIANFRAC", 0.9))
            )
            hessian_splits.append(misc.MergeDataset(_hessian_splits))
            hessian_names.append("e" + "".join(hessian.split(",")) + "_" + hessian_filter)

    else:
        hessian_splits, hessian_names = None, None

    if inf_args.mode.startswith("combin_"):
        start, end, top = [int(s) for s in inf_args.mode.split("_")[1:]]
        if end > len(good_checkpoints) + 1:
            gpuprint(f"{end} too big")
            end = len(good_checkpoints)

        for i in range(start, end):
            # random.shuffle(good_checkpoints)
            if top == 0:
                # random.shuffle(combinations_checkpoints)
                combinations_checkpoints = itertools.combinations(good_checkpoints, i)
            elif os.environ.get("MEMORY"):
                _combinations_all = list(itertools.combinations(good_checkpoints, i))
                random.shuffle(_combinations_all)
                combinations_checkpoints = _combinations_all[:top]
            else:
                combinations_checkpoints = [misc.random_combination(good_checkpoints, i) for _ in range(top)]

            for sub_good_checkpoints in combinations_checkpoints:
                if os.environ.get("DEBUG", "0") != "0":
                    ood_results = {}
                else:
                    ood_results = get_results_for_checkpoints(
                        sub_good_checkpoints, dataset, inf_args, ood_names, ood_splits, hessian_names,
                        hessian_splits, device
                    )
                index = -2 if os.environ.get("INFOLDER", "1") == "0" else -1
                ood_results["dirs"] = "_".join(
                    [checkpoint.split("/")[index] for checkpoint in sub_good_checkpoints]
                )
                ood_results["trials"] = "_".join(
                    [get_trial(checkpoint) for checkpoint in sub_good_checkpoints]
                )
                ood_results["i"] = str(i)
                gpuprint_results(inf_args, ood_results, i)

    elif inf_args.mode in ["all2"]:
        for sub_good_checkpoints in itertools.combinations(good_checkpoints, 2):
            gpuprint("")
            checkpoint0 = sub_good_checkpoints[0]
            checkpoint1 = sub_good_checkpoints[1]

            if -1 in inf_args.trial_seed and dict_checkpoints[checkpoint0][
                "trial_seed"] == dict_checkpoints[checkpoint1]["trial_seed"]:
                gpuprint(f"Skip f{sub_good_checkpoints} because same seeds")
                continue
            else:
                gpuprint(f"Process {sub_good_checkpoints}")

            if os.environ.get("DEBUG"):
                ood_results = {}
            else:
                ood_results = get_results_for_checkpoints(
                    sub_good_checkpoints, dataset, inf_args, ood_names, ood_splits, hessian_names,
                    hessian_splits, device
                )

            step0 = str(dict_checkpoints[checkpoint0]["step"])
            step1 = str(dict_checkpoints[checkpoint1]["step"])

            if step0 == step1 == "5000":
                ood_results["l"] = "l55"
            elif step0 == step1 == "3000":
                ood_results["l"] = "l33"
            elif dict_checkpoints[checkpoint0]["dir"] == dict_checkpoints[checkpoint1]["dir"]:
                ood_results["l"] = "ls"
            else:
                ood_results["l"] = "ld"

            # run_name = experiments_handler.get_run_name(inf_args.__dict__, {}, {})
            # experiments_handler.main_mlflow(
            #     run_name,
            #     ood_results,
            #     args=inf_args.__dict__,
            #     output_dir=None,
            #     hparams=None,
            #     version="soup"
            # )
            gpuprint_results(inf_args, ood_results, len(sub_good_checkpoints))
    elif inf_args.mode.startswith("iter_"):
        start, end = [int(s) for s in inf_args.mode.split("_")[1:]]
        if end > len(good_checkpoints) + 1:
            gpuprint(f"{end} too big")
            end = len(good_checkpoints)

        for i in range(start, end):
            sub_good_checkpoints = good_checkpoints[:i]
            if os.environ.get("DEBUG", "0") != "0":
                ood_results = {}
                gpuprint(i, sub_good_checkpoints)
            else:
                ood_results = get_results_for_checkpoints(
                    sub_good_checkpoints, dataset, inf_args, ood_names, ood_splits, hessian_names,
                    hessian_splits, device
                )
            ood_results["i"] = i - 1
            process_line_iter(ood_results, inf_args)
            gpuprint_results(inf_args, ood_results, i)

    elif inf_args.mode.startswith("iterg_"):
        start, end = [int(s) for s in inf_args.mode.split("_")[2:]]
        if end > len(good_checkpoints) + 1:
            gpuprint(f"{end} too big")
            end = len(good_checkpoints)
        good_indexes = []
        best_result = - float("inf")
        keymetric = inf_args.mode.split("_")[1].replace("-", "_")
        for i in range(start , end):
            i = i - 1
            good_indexes.append(i)
            sub_good_checkpoints = [good_checkpoints[index] for index in good_indexes]
            if os.environ.get("DEBUG", "0") != "0":
                ood_results = {keymetric: random.random()}
                gpuprint(i, sub_good_checkpoints)
            else:
                ood_results = get_results_for_checkpoints(
                    sub_good_checkpoints, dataset, inf_args, ood_names, ood_splits, hessian_names,
                    hessian_splits, device
                )
            ood_results["i"] = i
            process_line_iter(ood_results, inf_args)
            gpuprint_results(inf_args, ood_results, i)
            if "acc" in keymetric:
                new_result = ood_results[keymetric]
            else:
                new_result = - ood_results[keymetric]

            if new_result >= best_result:
                best_result = new_result
                gpuprint(f"Accepting index {i}")
            else:
                good_indexes.pop(-1)
                gpuprint(f"Skipping index {i}")

    elif inf_args.mode in ["", "ens"]:
        ood_results = get_results_for_checkpoints(
            good_checkpoints, dataset, inf_args, ood_names, ood_splits, hessian_names,
            hessian_splits, device
        )
        gpuprint_results(inf_args, ood_results, len(good_checkpoints))
    else:
        raise ValueError(inf_args.mode)


def process_line_iter(ood_results, inf_args):
    ood_results["dirs"] = ",".join(
        [output_dir.split("/")[-1] for output_dir in inf_args.output_dir.split(",")]
    )

    ood_results["trial_seed"] = ",".join([str(x) for x in inf_args.trial_seed])
    if inf_args.algorithm != "":
        ood_results["algo"] = inf_args.algorithm
    if os.environ.get("SWAMEMBER"):
        ood_results["swamember"] = os.environ.get("SWAMEMBER")

    # if "train" in inf_args.ood_data:
    #     ood_results["out_acc_soup"] = np.mean(
    #         [value for key, value in ood_results.items() if key.endswith("_out_acc_soup")]
    #     )
    #     for key in list(ood_results.keys()):
    #         if key.endswith("_out_acc_soup"):
    #             del ood_results[key]

    #     if inf_args.t_scaled:
    #         ood_results["out_ece_soup"] = np.mean(
    #             [value for key, value in ood_results.items() if key.endswith("_out_ece_soup")]
    #         )
    #         for key in list(ood_results.keys()):
    #             if key.endswith("_out_ece_soup"):
    #                 del ood_results[key]

    #     if inf_args.t_scaled.startswith("temp"):
    #         ood_results["out_ecetemp_soup"] = np.mean(
    #             [value for key, value in ood_results.items() if key.endswith("_out_ecetemp_soup")]
    #         )
    #         for key in list(ood_results.keys()):
    #             if key.endswith("_out_ecetemp_soup"):
    #                 del ood_results[key]

    return ood_results


def _get_args():
    parser = argparse.ArgumentParser(description='Domain generalization')

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--test_envs', type=int, nargs='+')
    parser.add_argument(
        '--trial_seed',
        type=str,
        default="-1",
        help='Trial number (used for seeding split_dataset and random_hparams).'
    )
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--data_dir', type=str, default="default")

    # select which checkpoints
    parser.add_argument('--mode', type=str, default="")  # "" or "all",
    parser.add_argument('--selection_strategy', type=str, default="")  # "greedy", "random", "zipf""
    parser.add_argument(
        '--cluster',
        type=str,
        default=[],
        nargs='+',
    )  # algorithm trial_seed
    parser.add_argument(
        '--regexes',
        type=str,
        default=[],
        nargs='+',
    )  # "soup_soupswa", "net0_net1"
    parser.add_argument('--ood_data', type=str, default="test")  # "test" or "train"

    parser.add_argument('--criteriontopk', type=str, default="acc_net")
    parser.add_argument('--topk', type=int, default=0)
    parser.add_argument('--selection_data', type=str, default="train")  # or "oracle"
    parser.add_argument('--zipf_a', type=float, default=3.)

    parser.add_argument('--algorithm', type=str, default="")
    parser.add_argument('--t_scaled', type=str, default="")
    parser.add_argument('--do_ens', type=str, default="")

    inf_args = parser.parse_args()
    if inf_args.data_dir == "default":
        if "DATA" in os.environ:
            inf_args.data_dir = os.path.join(os.environ["DATA"], "data/domainbed/")
        else:
            inf_args.data_dir = "domainbed/data"
    if inf_args.do_ens == "1":
        inf_args.do_ens = ["swa", "net"]
    elif inf_args.do_ens in ["0", ""]:
        inf_args.do_ens = []
    else:
        inf_args.do_ens = inf_args.do_ens.split(",")

    inf_args.trial_seed = [int(t) for t in inf_args.trial_seed.split(",")]
    return inf_args


def create_splits(inf_args, dataset, dict_env_to_filter, trial_seed=None, holdout_fraction=None):
    splits = []
    names = []
    assert len(dict_env_to_filter)

    for env_i, env in enumerate(dataset):
        filter = None
        if "test" in dict_env_to_filter and env_i in inf_args.test_envs:
            filter = dict_env_to_filter["test"]
        elif "train" in dict_env_to_filter and env_i not in inf_args.test_envs:
            filter = dict_env_to_filter["train"]
        elif str(env_i) in dict_env_to_filter:
            filter = dict_env_to_filter[str(env_i)]

        if filter is None:
            continue

        if filter == "full":
            splits.append(env)
            names.append('e{}'.format(env_i))
        else:
            holdout_fraction = (
                holdout_fraction if holdout_fraction is not None else inf_args.holdout_fraction
            )
            out_, in_ = misc.split_dataset(
                env, int(len(env) * holdout_fraction), misc.seed_hash(trial_seed, env_i)
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


def get_score_run(results, criteriontopk, test_envs, selection_data="train"):
    if not results:
        return 0.
    if criteriontopk in ["none", "0"]:
        return 0.
    if criteriontopk in ["step"]:
        return results[criteriontopk]
    if criteriontopk in ["minus_step"]:
        return -results[criteriontopk.split("_")[-1]]

    if criteriontopk.startswith("acc"):
        criteriontopk = f"Accuracies/{criteriontopk}"
    elif criteriontopk.startswith("ece"):
        criteriontopk = f"Calibration/{criteriontopk}"
    else:
        raise ValueError(f"Unknown criterion {criteriontopk}")

    val_env_keys = []
    for i in itertools.count():
        acc_key = f'env{i}_out_{criteriontopk}'
        if acc_key in results:
            if selection_data == "train":
                if i not in test_envs:
                    val_env_keys.append(acc_key)
            elif selection_data == "test":
                if i in test_envs:
                    val_env_keys.append(acc_key)
            else:
                raise ValueError(selection_data)
        else:
            break
    assert i > 0
    return np.mean([results[key] for key in val_env_keys])


def gpuprintv(s, v=True):
    if v:
        gpuprint(s)



def get_trial(checkpoint):
    save_dict = torch.load(os.path.join(checkpoint, "model.pkl"))
    train_args = NameSpace(save_dict["args"])
    return str(train_args.trial_seed)


def find_checkpoints(inf_args, verbose=False):
    checkpoints = [
        os.path.join(output_dir, path)
        for output_dir in inf_args.output_dir.split(",")
        for path in os.listdir(output_dir)
    ]
    if os.environ.get("INFOLDER", "1") != "0":
        checkpoints = [
            os.path.join(checkpoint, path)
            for checkpoint in checkpoints
            if os.path.isdir(checkpoint) for path in os.listdir(checkpoint)
            if path not in os.environ.get("SKIPSTEPS", "bestswa").split("_")
        ]

    found_checkpoints_per_cluster = {}
    dict_checkpoints = {}
    set_unique_key = set()
    for folder in checkpoints:
        if not os.path.isdir(folder):
            continue
        name_folder = os.path.split(folder)[-1]
        model_path = os.path.join(folder, "model.pkl")
        if not os.path.exists(model_path):
            gpuprintv(f"absent: {name_folder}", verbose)
            continue
        save_dict = torch.load(model_path)
        train_args = NameSpace(save_dict["args"])
        unique_key = str(train_args.algorithm) + "_" + str(save_dict["model_hparams"]["lr"]) + "_" + str(save_dict["model_hparams"]["resnet_dropout"])

        if train_args.dataset != inf_args.dataset:
            gpuprintv(f"bad dataset: {name_folder}", verbose)
            continue
        if train_args.test_envs != inf_args.test_envs:
            gpuprintv(f"bad test env: {name_folder}", verbose)
            continue
        if inf_args.algorithm != "" and train_args.algorithm not in inf_args.algorithm.split(","):
            gpuprintv(f"bad algorithm: {name_folder}", verbose)
            continue
        if (train_args.trial_seed not in inf_args.trial_seed and -1 not in inf_args.trial_seed):
            gpuprintv(f"bad trial seed: {name_folder}", verbose)
            continue
        if train_args.holdout_fraction != inf_args.holdout_fraction:
            gpuprintv(f"Warning different holdout fraction: {name_folder} but keep", verbose)
        if os.environ.get("WEIGHT_DECAY") and save_dict["model_hparams"]["weight_decay"] != float(os.environ.get("WEIGHT_DECAY")):
            gpuprintv(f"Bad weight decay: {save_dict['model_hparams']['weight_decay']} in {name_folder}", True)
            continue

        if os.environ.get("UNIQ", "0") != "0":
            if unique_key in set_unique_key:
                gpuprint(f"Skip {folder} of {unique_key}")
                continue
            else:
                set_unique_key.add(unique_key)

        gpuprintv(f"found: {name_folder}", verbose)
        run_results = json.loads(save_dict.get("results", ""))
        score_folder = get_score_run(
            run_results, criteriontopk=inf_args.criteriontopk, test_envs=inf_args.test_envs, selection_data=inf_args.selection_data
        )
        if os.environ.get("STEPS"):
            step = run_results.get("step", 5000)
            if os.environ.get("STEPS").startswith("mod"):
                if name_folder == "best":
                    print("Skip best")
                    continue
                if int(step) % int(os.environ.get("STEPS")[3:]) != 0:
                    gpuprintv(f"bad step {step} for: {name_folder}", verbose)
                    continue
            else:
                if str(step) not in os.environ.get("STEPS").split("_"):
                    continue

        train_args.__dict__["dir"] = os.path.split(os.path.split(folder)[0])[-1]
        cluster = "|".join([str(train_args.__dict__[cluster]) for cluster in inf_args.cluster])
        if cluster not in found_checkpoints_per_cluster:
            found_checkpoints_per_cluster[cluster] = {}
        found_checkpoints_per_cluster[cluster][folder] = score_folder
        dict_checkpoints[folder] = train_args.__dict__
        dict_checkpoints[folder]["step"] = run_results.get("step", 5000)
        dict_checkpoints[folder]["unique_key"] = unique_key


    if len(found_checkpoints_per_cluster) == 0:
        raise ValueError(f"No checkpoints found for: {inf_args}")
        return []
    sorted_checkpoints_per_cluster = {
        cluster: sorted(found_checkpoints.keys(), key=lambda x: found_checkpoints[x], reverse=True)
        for cluster, found_checkpoints in found_checkpoints_per_cluster.items()
    }
    gpuprintv(sorted_checkpoints_per_cluster, verbose)

    # if os.environ.get("UNIQ", "0") != "0":
    #     filtered_dict = {}
    #     for cluster, sorted_checkpoints in sorted_checkpoints_per_cluster.items():
    #         set_unique_key = set()
    #         filtered_dict[cluster] = []
    #         for checkpoint in sorted_checkpoints:
    #             unique_key = dict_checkpoints[checkpoint]["unique_key"]
    #             if unique_key in set_unique_key:
    #                 gpuprint(f"Skip {checkpoint} of {unique_key}")
    #             else:
    #                 filtered_dict[cluster].append(sorted_checkpoints)
    #                 set_unique_key.add(unique_key)
    #     sorted_checkpoints_per_cluster = filtered_dict
    #     gpuprintv(sorted_checkpoints_per_cluster, True)

    dict_checkpoints_to_score = {
        checkpoint: found_checkpoints_per_cluster[cluster][checkpoint]
        for cluster, found_checkpoints in found_checkpoints_per_cluster.items()
        for checkpoint in found_checkpoints}

    return sorted_checkpoints_per_cluster, dict_checkpoints, dict_checkpoints_to_score


def file_with_weights(folder):
    filename = os.path.join(folder, "model.pkl")
    filename_heavy = os.path.join(folder, "model_with_weights.pkl")
    if os.path.exists(filename_heavy):
        filename = filename_heavy
    else:
        # gpuprint(f"missing {filename_heavy}")
        assert os.path.exists(filename)
    return filename


def get_good_checkpoints(sorted_checkpoints_per_cluster, inf_args, dataset, device, dict_checkpoints_to_score):
    good_checkpoints = []
    for cluster, found_checkpoints in sorted_checkpoints_per_cluster.items():
        gpuprint(f"Exploring cluster: {cluster} with {len(found_checkpoints)} checkpoints")
        if inf_args.selection_strategy == "greedy":
            gpuprint(f"Select from greedy")
            if "trial_seed" in inf_args.cluster:
                assert inf_args.selection_data == "train"
                trial_seed = int(cluster.split("|")[inf_args.cluster.index("trial_seed")])
            else:
                trial_seed = inf_args.trial_seed[0]
            val_splits, val_names = create_splits(
                inf_args,
                dataset,
                dict_env_to_filter={"train" if inf_args.selection_data == "train" else "test": "out"},
                trial_seed=trial_seed
            )
            cluster_good_checkpoints = get_greedy_checkpoints(
                found_checkpoints, dataset, inf_args, val_names, val_splits, device
            )
        elif inf_args.selection_strategy == "zipf":
            gpuprint(f"Select from zipf")
            cluster_good_checkpoints = get_from_zipf(
                found_checkpoints, inf_args.topk, a=inf_args.zipf_a
            )
        elif inf_args.selection_strategy in ["random"]:
            gpuprint(f"Select random")
            topk = min(len(found_checkpoints), inf_args.topk)
            rand_nums = sorted(random.sample(range(len(found_checkpoints)), topk))
            cluster_good_checkpoints = [found_checkpoints[i] for i in rand_nums]
        elif inf_args.topk != 0:
            gpuprint(f"Select best")
            cluster_good_checkpoints = found_checkpoints[:inf_args.topk]
        else:
            gpuprint(f"Select all")
            cluster_good_checkpoints = found_checkpoints[:]
        gpuprint(f"Select {len(cluster_good_checkpoints)}/{len(found_checkpoints)} checkpoints")
        good_checkpoints.append(cluster_good_checkpoints)

    if len(good_checkpoints) == 1:
        return good_checkpoints[0]
    return zip_unequal(good_checkpoints, dict_checkpoints_to_score)


def zip_unequal(good_checkpoints, dict_checkpoints_to_score):
    max_len = max([len(l) for l in good_checkpoints])
    outs = []
    for i in range(max_len):
        outs_at_i = {}
        for l in good_checkpoints:
            if i < len(l):
                outs_at_i[l[i]] = dict_checkpoints_to_score[l[i]]
        reverse = (os.environ.get("REVERSEZIP", "True") == "True")
        outs.extend(sorted(outs_at_i.keys(), key=lambda x: outs_at_i[x], reverse=reverse))
    return outs


def get_greedy_checkpoints(found_checkpoints, dataset, inf_args, val_names, val_splits, device):

    ens_algorithm_class = algorithms_inference.get_algorithm_class("Soup")
    ens_algorithm = ens_algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(inf_args.test_envs),
        t_scaled=False,
        regexes=[],
        do_ens=[]
    )
    best_results = {}
    good_nums = []
    for num, folder in enumerate(found_checkpoints):
        # gpuprint(f"Ingredient {num} from folder: {os.path.split(folder)[-1]}")

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
            # gpuprint(f"Inference at {name} at num {num}")
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

        # gpuprint(f"Val results for {inf_args} at {num}")
        # results_keys = sorted(val_results.keys())
        # misc.gpuprint_row([key.split("/")[-1] for key in results_keys], colwidth=15, latex=True)
        # misc.gpuprint_row([val_results[key] for key in results_keys], colwidth=15, latex=True)

        for key in val_results:
            if not key.startswith("Accuracies"):
                continue
            if val_results[key] > best_results.get(key, -float("inf")):
                if key == "Accuracies/acc_soup":
                    good_nums.append(num)
                best_results[key] = val_results[key]

        if num not in good_nums:
            # ens_algorithm.delete_last()
            gpuprint(f"Stop at num {num}")
            break
        else:
            gpuprint(f"Add num {num}")
        # gpuprint("")

    gpuprint(f"Best OOD results for {inf_args} with {len(good_nums)} checkpoints")
    gpuprint(best_results)
    return [found_checkpoints[num] for num in good_nums]


def get_from_zipf(found_checkpoints, topk, a=3):
    n = len(found_checkpoints)
    nums = set([])
    while len(nums) != topk:
        z = np.random.zipf(a, 1)[0]
        if z < n:
            nums.add(z)
    return [checkpoint for i, checkpoint in enumerate(found_checkpoints) if i in nums]



def get_results_for_checkpoints(
    good_checkpoints, dataset, inf_args, ood_names, ood_splits, hessian_names, hessian_splits,
    device
):
    ens_algorithm_class = algorithms_inference.get_algorithm_class("Soup")
    ens_algorithm = ens_algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(inf_args.test_envs),
        t_scaled=inf_args.t_scaled,
        regexes=inf_args.regexes,
        do_ens=inf_args.do_ens
    )
    for folder in good_checkpoints:
        gpuprint(f"Ingredient from folder: {folder}")
        save_dict = torch.load(file_with_weights(folder))
        train_args = NameSpace(save_dict["args"])

        # load model
        algorithm_class = algorithms_inference.get_algorithm_class(train_args.algorithm)
        algorithm = algorithm_class(
            dataset.input_shape, dataset.num_classes,
            len(dataset) - len(inf_args.test_envs),
            save_dict["model_hparams"]
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
        FastDataLoader(dataset=split, batch_size=64, num_workers=dataset.N_WORKERS)
        for split in ood_splits
    ]

    ood_evals = zip(ood_names, ood_loaders)
    ood_results = {}
    for name, loader in ood_evals:
        # gpuprint(f"Inference at {name}")
        update_temperature = (name == "train_out" and inf_args.t_scaled == "temp_out")
        if name == "train_full":
            ens_algorithm.do_ens = []
        else:
            ens_algorithm.do_ens = inf_args.do_ens

        results = ens_algorithm.accuracy(
            loader, device, compute_trace=True,
            update_temperature=update_temperature
            )
        # gpuprint(results)
        for key in results:
            clean_key = key.split("/")[-1]
            ood_results[name + "_" + clean_key] = results[key]

    if hessian_splits is not None:
        hessian_loaders = [
            FastDataLoader(
                dataset=split,
                batch_size=int(os.environ.get("HESSIANBS", 64)),
                num_workers=dataset.N_WORKERS
            ) for split in hessian_splits
        ]
        hessian_evals = zip(hessian_names, hessian_loaders)
        for i, (name, loader) in enumerate(hessian_evals):
            gpuprint(f"Hessian at {name} for {len(loader)} batches")
            hessian_results = ens_algorithm.compute_hessian(loader)
            for key in hessian_results:
                clean_key = key.split("/")[-1]
                ood_results[name + "_" + clean_key] = hessian_results[key]
                # ood_results[clean_key] = ood_results.get(
                #     clean_key, 0
                # ) + hessian_results[key] / len(hessian_names)
    ood_results["gpu"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    ood_results["length"] = len(good_checkpoints)
    return ood_results


def gpuprint_results(inf_args, ood_results, len_):
    ood_results_keys = sorted(ood_results.keys())
    gpuprint(
        f"OOD results for {inf_args} with {len_}"
    )
    misc.print_rows(
        row1=ood_results_keys,
        row2=[ood_results[key] for key in ood_results_keys],
    )


if __name__ == "__main__":
    main()
