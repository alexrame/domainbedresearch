# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
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
    parser.add_argument(
        '--task',
        type=str,
        default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"]
    )
    parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')
    parser.add_argument('--sweep_id', type=str, help='', default="single")
    parser.add_argument("--hp", nargs=2, action="append")
    parser.add_argument(
        '--hparams_seed',
        type=int,
        default=0,
        help='Seed for random hparams (0 means "default hparams")'
    )
    parser.add_argument(
        '--trial_seed',
        type=int,
        default=0,
        help='Trial number (used for seeding split_dataset and random_hparams).'
    )
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument(
        '--steps', type=int, default=None, help='Number of steps. Default is dataset-dependent.'
    )
    parser.add_argument(
        '--checkpoint_freq',
        type=int,
        default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.'
    )
    parser.add_argument('--save_model_every_checkpoint', action='store_true')

    parser.add_argument('--test_envs', type=int, nargs='+')
    parser.add_argument('--output_dir', type=str, default="default+name")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument(
        '--uda_holdout_fraction',
        type=float,
        default=0,
        help="For domain adaptation, % of test to use unlabeled for training."
    )
    parser.add_argument('--skip_model_save', action='store_true')
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0

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

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(
            args.algorithm, args.dataset, misc.seed_hash(args.hparams_seed, args.trial_seed)
        )
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    if args.hp:
        hp = {key: yaml.safe_load(val) for (key, val) in args.hp}
        for key in hp:
            if key not in hparams:
                print(key)
                print(hparams)
                raise ValueError("key", key, "does not exist")
        print("Update param: ", hp)

        hparams.update(hp)
    else:
        hp = {}

    run_name = experiments_handler.get_run_name(args.__dict__, hparams=hparams, hp=hp)

    if args.data_dir == "default":
        if "DATA" in os.environ:
            args.data_dir = os.path.join(os.environ["DATA"], "data/domainbed/")
        else:
            args.data_dir = "domainbed/data"

    if args.output_dir == "default+name":
        if "DATA" in os.environ:
            args.output_dir = os.path.join(
                os.environ["DATA"], f"experiments/domainbed/singleruns/{args.dataset}", run_name
            )
        else:
            args.output_dir = os.path.join(f"logs/singleruns/{args.dataset}", run_name)

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    writer = SummaryWriter(log_dir=args.output_dir)
    # writer.add_hparams(hparams , dict())

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset_class = vars(datasets)[args.dataset]
        if args.test_envs is None:
            args.test_envs = dataset_class.TEST_ENVS
        dataset = dataset_class(args.data_dir, args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selection method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discarded at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)

    for env_i, env in enumerate(dataset):
        uda = []

        if dataset_class.CUSTOM_DATASET:
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

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    print("Train Envs:", [i for (i, _) in enumerate(in_splits) if i not in args.test_envs])

    train_loaders = [
        InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=hparams['batch_size'],
            num_workers=dataset.N_WORKERS
        ) for i, (env, env_weights) in enumerate(in_splits) if i not in args.test_envs
    ]

    uda_loaders = [
        InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=hparams['batch_size'],
            num_workers=dataset.N_WORKERS
        ) for i, (env, env_weights) in enumerate(uda_splits) if i in args.test_envs
    ]

    eval_loaders = [
        FastDataLoader(dataset=env, batch_size=64, num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)
    ]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i) for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i) for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i) for i in range(len(uda_splits))]

    algorithm: Algorithm = algorithm_class(
        dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams
    )

    algorithm.to(device)

    if hparams.get("lrdecay"):
        scheduler = ExponentialLR(algorithm.optimizer, gamma=float(hparams.get("lrdecay")))
    else:
        scheduler = None

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    steps_per_epoch = min(
        [
            len(env) / hparams['batch_size']
            for i, (env, _) in enumerate(in_splits)
            if i not in args.test_envs
        ]
    )
    # print([len(env)/hparams['batch_size'] for i, (env,_) in enumerate(in_splits) if i not in args.test_envs])
    # print([len(env) for i, (env,_) in enumerate(in_splits) if i not in args.test_envs])
    print(
        f"n_steps: {n_steps} / n_epochs: {n_steps / steps_per_epoch} / steps_per_epoch: {steps_per_epoch} / checkpoints: {n_steps / checkpoint_freq}"
    )

    def save_checkpoint(filename, results, filename_heavy=None, **kwargs):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "results": results,
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_hparams": hparams,
        }
        save_dict.update(kwargs)
        file_path = os.path.join(args.output_dir, filename)
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(save_dict, file_path)
        print(f"Model saved to: {file_path}")
        if filename_heavy:
            save_dict["model_dict"] = algorithm.cpu().state_dict()
            if algorithm.hparams.get("swa"):
                if algorithm.swas is not None:
                    for i, swa in enumerate(algorithm.swas):
                        save_dict[f"swa{i}_dict"] = swa.network_swa.cpu().state_dict()
                if algorithm.swa is not None:
                    save_dict["swa_dict"] = algorithm.swa.network_swa.cpu().state_dict()
            file_path_heavy = os.path.join(args.output_dir, filename_heavy)
            directory = os.path.dirname(file_path_heavy)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(save_dict, file_path_heavy)

    def get_score(results):
        import itertools
        criteriontopk = f"Accuracies/acc_net"
        val_env_keys = []
        for i in itertools.count():
            acc_key = f'env{i}_out_{criteriontopk}'
            if acc_key in results:
                if i not in args.test_envs:
                    val_env_keys.append(acc_key)
            else:
                break
        assert i > 0
        return np.mean([results[key] for key in val_env_keys])

    last_results_keys = None
    metrics = {}
    results_end = {}
    best_score = -float("inf")


    for step in tqdm(range(start_step, n_steps)):
        step_start_time = time.time()
        batches = [b for b in next(train_minibatches_iterator)]
        if args.task == "domain_adaptation":
            uda_batch = next(uda_minibatches_iterator)
        else:
            uda_batch = None
        minibatches_device = [(x.to(device), y.to(device)) for x, y in batches]
        if args.task == "domain_adaptation":
            uda_device = [x.to(device) for x, _ in uda_batch]
        else:
            uda_device = None

        step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        do_metrics = (step % checkpoint_freq == 0) or step == n_steps - 1
        if os.environ.get('DO_LAST10'):
            do_metrics |= step >= n_steps - 10

        if os.environ.get("STEPS"):
            if os.environ.get("STEPS") == "all":
                do_metrics |= step <= 100 and step % 10 == 0
                do_metrics |= step <= 10 and step % 2 == 0
            elif os.environ.get("STEPS").startswith("mod"):
                do_metrics |= int(step) % int(os.environ.get("STEPS")[3:]) == 0
            else:
                do_metrics |= step in [int(s) for s in os.environ.get("STEPS").split("_")]

        if do_metrics:
            results = {'step': step}
            if scheduler is not None:
                results["lr"] = scheduler.get_lr()

            for key, val in checkpoint_vals.items():
                try:
                    results[key] = np.mean(val)
                    writer.add_scalar("Metrics/" + key, results[key], step)
                except Exception as exc:
                    print(exc)
                    print(key, val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                if hasattr(algorithm, "accuracy"):
                    if step == n_steps - 1 and os.environ.get("HESSIAN", "0") != "0":
                        traced_envs = [args.test_envs[0], args.test_envs[0] +
                                       1] if args.test_envs[0] != 3 else [1, 3]
                        compute_trace = any([("env" + str(env)) in name for env in traced_envs])
                    else:
                        compute_trace = False
                    update_temperature = False
                    # name in [
                    #     'env{}_out'.format(i)
                    #     for i in range(len(out_splits))
                    #     if i not in args.test_envs
                    # ]
                    acc = algorithm.accuracy(
                        loader, device, compute_trace, update_temperature=update_temperature
                    )
                    if compute_trace:
                        try:
                            acc.update(algorithm.compute_hessian(loader))
                        except Exception as exc:
                            print("Failure during Hessian computation")
                            print(exc)
                else:
                    acc = misc.accuracy(algorithm, loader, weights, device)
                for key in acc:
                    results[name + f'_{key}'] = acc[key]
                    if "/" not in key:
                        tb_name = f'{name}_{key}'
                    else:
                        tb_name = f'{key.split("/")[0]}/{name}_{key.split("/")[1]}'
                    writer.add_scalar(tb_name, acc[key], step)

            results_keys = sorted(results.keys())
            printed_keys = [key for key in results_keys if "diversity" not in key.lower()]
            if results_keys != last_results_keys:
                misc.print_row([key.split("/")[-1] for key in printed_keys], colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in printed_keys], colwidth=12)

            if 0 < n_steps - step <= 10 and os.environ.get('DO_LAST10'):
                print(f"Update results_end at step {step}")
                if len(results_end) != 0:
                    if len(results_end) != len(results):
                        print(results_end)
                        print(results)
                for key in results:
                    results_end[key] = results_end.get(key, 0.) + results[key] / 10.
                misc.print_row([results_end[key] for key in printed_keys], colwidth=12)

            results.update({'hparams': hparams, 'args': vars(args)})

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                results_dumpable = {
                    key: value for key, value in results.items() if misc.is_dumpable(value)
                }
                f.write(json.dumps(results_dumpable, sort_keys=True) + "\n")
            current_score = get_score(results)
            if current_score > best_score:
                best_score = current_score
                print("Saving new best score at epoch")
                save_checkpoint(
                    'best/model.pkl',
                    results=json.dumps(results_dumpable, sort_keys=True),
                    filename_heavy=f'{step}/model_with_weights.pkl'
                )
                algorithm.to(device)

            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            save_epoch = args.save_model_every_checkpoint
            if os.environ.get("STEPS"):
                if os.environ.get("STEPS") == "all":
                    save_epoch = True
                elif os.environ.get("STEPS").startswith("mod"):
                    save_epoch |= int(step) % int(os.environ.get("STEPS")[3:]) == 0
                else:
                    save_epoch |= step in [
                        int(s) for s in os.environ.get("STEPS").split("_")
                    ]
            if save_epoch:
                save_checkpoint(
                    f'{step}/model.pkl',
                    results=json.dumps(results_dumpable, sort_keys=True),
                    filename_heavy=f'{step}/model_with_weights.pkl'
                )
                algorithm.to(device)

            for key, value in algorithm.get_tb_dict().items():
                writer.add_scalar(f'General/{key}', value, step)

        if scheduler is not None:
            scheduler.step()

    if hasattr(dataset, "after_training"):
        dataset.after_training(algorithm, args.output_dir, device=device)

    results_dumpable = {key: value for key, value in results.items() if misc.is_dumpable(value)}
    save_checkpoint(
        'model.pkl',
        results=json.dumps(results_dumpable, sort_keys=True),
        filename_heavy='model_with_weights.pkl'
    )
    algorithm._save_network_for_future()

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')

    if not os.environ.get('DO_LAST10'):
        results_end = results
    metrics.update({k: v for k, v in results_end.items() if k not in ["hparams", "args"]})

    # metrics.update({"end" + str(k): v for k, v in results_end.items() if k not in ["hparams", "args", "step", "epoch", "lr"]})
    experiments_handler.main_mlflow(
        run_name, metrics, args=args.__dict__, output_dir=args.output_dir, hparams=hparams
    )


if __name__ == "__main__":
    main()
