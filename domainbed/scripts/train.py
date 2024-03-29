# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
import statistics
import traceback
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
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--sweep_id', type=str, help='', default="single")
    parser.add_argument("--hp", nargs=2, action="append")
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+')
    parser.add_argument(
        '--output_dir', type=str, default="default+name"
    )
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None


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
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    if args.hp:
        hp = {
            key: yaml.safe_load(val) for (key, val) in args.hp
        }
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
            args.data_dir = os.path.join(os.environ["DATA"],"data/domainbed/")
        else:
            args.data_dir = "domainbed/data"

    if args.output_dir == "default+name":
        if "DATA" in os.environ:
            args.output_dir = os.path.join(
                os.environ["DATA"],
                f"experiments/domainbed/singleruns/{args.dataset}",
                run_name
            )
        else:
            args.output_dir = os.path.join(
                f"logs/singleruns/{args.dataset}",
                run_name
            )


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
        dataset = dataset_class(args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)

    for env_i, env in enumerate(dataset):
        uda = []

        if not algorithm_class.CUSTOM_FORWARD and dataset_class.CUSTOM_DATASET:
            env = misc.CustomToRegularDataset(env)

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

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

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    algorithm: Algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)
    try:
        algorithm.member_diversifier.to(device)
    except:
        pass
    try:
        algorithm.member_diversifier.q = algorithm.member_diversifier.q.to(device)
    except:
        pass
    try:
        algorithm.mav.network_mav.to(device)
    except:
        pass
    try:
        for mav in algorithm.mavs:
            mav.network_mav.to(device)
    except:
        pass

    if hparams.get("lrdecay"):
        scheduler = ExponentialLR(
            algorithm.optimizer,
            gamma=float(hparams.get("lrdecay")))
    else:
        scheduler = None

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for i, (env,_) in enumerate(in_splits) if i not in args.test_envs])
    # print([len(env)/hparams['batch_size'] for i, (env,_) in enumerate(in_splits) if i not in args.test_envs])
    # print([len(env) for i, (env,_) in enumerate(in_splits) if i not in args.test_envs])
    # print(steps_per_epoch)

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.cpu().state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    last_results_keys = None
    metrics = {}
    for step in tqdm(range(start_step, n_steps)):
        step_start_time = time.time()
        batches = [b for b in next(train_minibatches_iterator)]
        if args.task == "domain_adaptation":
            uda_batch = next(uda_minibatches_iterator)
        else:
            uda_batch = None
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in batches]
        if args.task == "domain_adaptation":
            uda_device = [x.to(device)
                for x,_ in uda_batch]
        else:
            uda_device = None
        step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1) or (step < 0 and step % 2 ==0):

            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }
            if scheduler is not None:
                results["lr"] = scheduler.get_lr()

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)
                writer.add_scalar("Metrics/" + key, results[key], step)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                # tqdm.write("eval "+name)
                # begin = time.time()
                if hasattr(algorithm, "accuracy"):
                    acc = algorithm.accuracy(loader, device)
                else:
                    acc = misc.accuracy(algorithm, loader, weights, device)
                # tqdm.write("time:" + str(time.time() - begin))
                for key in acc:
                    results[name + f'_{key}'] = acc[key]
                    if "/" not in key:
                        tb_name = f'{name}_{key}'
                    else:
                        tb_name = f'{key.split("/")[0]}/{name}_{key.split("/")[1]}'
                    writer.add_scalar(tb_name, acc[key], step)

            results_keys = sorted(results.keys())
            printed_keys = [
                key for key in results_keys
                if "Diversity" not in key.lower()]
            if results_keys != last_results_keys:
                misc.print_row([key.split("/")[-1] for key in printed_keys], colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in printed_keys],
                colwidth=12)

            if len(args.test_envs) == 1:
                key = "env" + str(args.test_envs[0]) + "_in_acc"
                if key in results:
                    train_acc = statistics.mean([
                            results["env" + str(i) + "_out_acc"]
                            for i, (env, env_weights) in enumerate(in_splits)
                            if i not in args.test_envs
                        ])
                    if step == 5000:
                        metrics["acc_5000"] = results.get(key, 0.)
                        metrics["trainacc_5000"] = train_acc

                    if results.get(key, 0.) > metrics.get("best_acc", 0.) and step / n_steps > 0.66:
                        metrics["best_acc"] = results.get(key, 0.)
                    key_val = "env" + str(args.test_envs[0]) + "_out_acc"

                    if results.get(key_val, 0.) > metrics.get("best_acc_val", 0.) and step / n_steps > 0.66:
                        metrics["best_acc_val"] = results.get(key_val, 0.)
                        metrics["acc_wrt_bestval"] = results.get(key, 0.)
                        metrics["trainacc_wrt_bestval"] = train_acc

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                try:
                    f.write(json.dumps(results, sort_keys=True) + "\n")
                except Exception as e:
                    def is_dumpable(value):
                        try:
                            json.dumps(value)
                        except:
                            return False
                        return True

                    results_dumpable = {
                        key: value for key, value in results.items() if is_dumpable(value)
                    }
                    results_nodumpable = {
                        key: value for key, value in results.items() if not is_dumpable(value)
                    }
                    # import pdb; pdb.set_trace()
                    print(e)
                    print(results_nodumpable)
                    f.write(json.dumps(results_dumpable, sort_keys=True) + "\n")

                    # f.write(json.dumps({key: results[key] for key in results_keys}, sort_keys=True) + "\n")


            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

            for key, value in algorithm.get_tb_dict().items():
                writer.add_scalar(f'General/{key}', value, step)

        if scheduler is not None:
            scheduler.step()

    if hasattr(dataset, "after_training"):
        dataset.after_training(algorithm, args.output_dir, device=device)

    save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')

    metrics.update({k: v for k, v in results.items() if k not in ["hparams", "args"]})
    experiments_handler.main_mlflow(
        run_name,
        metrics,
        args=args.__dict__,
        output_dir=args.output_dir,
        hparams=hparams,
    )



if __name__ == "__main__":
    main()
