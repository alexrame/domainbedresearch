
# scaling number of runs from 1 to M


/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm601shhps0406home

CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/sam203shhps0426home --mode iterg_train-out-acc-soup_1_60 --ood_data train,test --do_ens 0 --trial_seed 0 &
CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/sam203shhps0426home --mode iterg_train-out-acc-soup_1_60 --ood_data train,test --do_ens 0 --trial_seed 1 &
CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/sam203shhps0426home --mode iterg_train-out-acc-soup_1_60 --ood_data train,test --do_ens 0 --trial_seed 2 &
wait


home_erm601shhps0406_env0.slurm

/gpfsdswork/projects/rech/edr/utr15kn/slurmconfig/0406/combinhome0_erm601_1to60.slurm



# SAM

iterghome0_sam203shhps0426.slu


# ens table 1

INFOLDER=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --mode ens --do_ens net --dataset TerraIncognita --test_envs 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0424terra --trial_seed 0
INFOLDER=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --mode ens --do_ens net --dataset TerraIncognita --test_envs 1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0424terra --trial_seed 0
INFOLDER=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --mode ens --do_ens net --dataset TerraIncognita --test_envs 2 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0424terra --trial_seed 0
INFOLDER=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --mode ens --do_ens net --dataset TerraIncognita --test_envs 3 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0424terra --trial_seed 0

INFOLDER=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --mode ens --do_ens net --dataset TerraIncognita --test_envs 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0424terra --trial_seed 1
INFOLDER=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --mode ens --do_ens net --dataset TerraIncognita --test_envs 1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0424terra --trial_seed 1
INFOLDER=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --mode ens --do_ens net --dataset TerraIncognita --test_envs 2 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0424terra --trial_seed 1
INFOLDER=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --mode ens --do_ens net --dataset TerraIncognita --test_envs 3 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0424terra --trial_seed 1

INFOLDER=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --mode ens --do_ens net --dataset TerraIncognita --test_envs 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0424terra --trial_seed 2
INFOLDER=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --mode ens --do_ens net --dataset TerraIncognita --test_envs 1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0424terra --trial_seed 2
INFOLDER=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --mode ens --do_ens net --dataset TerraIncognita --test_envs 2 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0424terra --trial_seed 2
INFOLDER=1 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --mode ens --do_ens net --dataset TerraIncognita --test_envs 3 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm203shlphps0424terra --trial_seed 2
