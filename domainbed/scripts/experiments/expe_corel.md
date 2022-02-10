CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.train --data_dir=/data/rame/data/domainbed/ --algorithm COREL --dataset ColoredMNIST --test_env 2



CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.train --data_dir=/data/rame/data/domainbed/ --algorithm COREL --dataset ColoredMNIST --test_env 2 --hp method none


CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.sweep launch --data_dir=/data/rame/data/domainbed/ --output_dir=/local/rame/experiments/domainbed/ --command_launcher local --algorithms IGA --datasets ColoredMNIST --n_hparams 5 --n_trials 1 --single_test_envs




export LOGDIR=w:${ML}/65/8c1197c491fb4d3bb1ddbb61888a8baa/artifacts,0:${ML}/65/d002821463354a2e9e22f3b79a796d8d/artifacts
