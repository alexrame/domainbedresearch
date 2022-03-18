# commands to test

PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --test_envs 0 --trial_seed -1 --output_dir /data/rame/experiments/domainbed/erm66shhpeoa0317/ --cluster trial_seed --topk 1 --criteriontopk acc_net
CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.train --algorithm Ensembling --dataset ColoredMNIST --test_env 2 --hp swa 1
#


Ingredient from folder: d04d868e36efe8eabb47043317ff7545
Ingredient from folder: 4acdaab3d3b950235498f1f21c35fd04
Ingredient from folder: c277e5ff8eb4b7c4811158018b62ade2
Ingredient from folder: c4f3a5fc2f9ee2464ecd0a0079ecd51d
Ingredient from folder: 0583a640ee2afc0cd74c88540ba06bad
Ingredient from folder: 0636bccff11f57d695ae431ae6861884
Ingredient from folder: 481ce14213c190438e09806b5a64d3b6
Ingredient from folder: dacf9c2dcb4e7d8a88b8cc4fdb2a0b97
Inference at env0_in
acc_net      & acc_netm     & acc_soup     & acc_soupswa  & acc_swa      & acc_swam     & ece_net      & ece_netm     & ece_soup     & ece_soupswa  & ece_swa      & ece_swam     & net01qstat   & net01ratio   & swa0swa1qsta & swa0swa1rati & swanetqstat  & swanetratio
0.6421215242 & 0.5690653965 & 0.6694129763 & 0.6740473738 & 0.6683831102 & 0.6583419156 & 18.791041562 & 25.566174227 & 17.355055594 & 15.219724199 & 16.785984606 & 17.533813043 & 0.9011590137 & 0.5555555556 & 0.9835111405 & 0.2619863014 & 0.9904405256 & 0.2132231405 \\






good: dacf9c2dcb4e7d8a88b8cc4fdb2a0b97
Ingredient from folder: f8e0a930e734a93fb34914272261972b
Ingredient from folder: 18546b99a5243f22e4a499eb06b80176
Ingredient from folder: 295920c39b0338962b0a2ae7518b279f
Ingredient from folder: d04d868e36efe8eabb47043317ff7545
Ingredient from folder: 4acdaab3d3b950235498f1f21c35fd04
Ingredient from folder: 4f04c25ea1b80d0f611497e4650f120f
Ingredient from folder: 97f546a048ad46a5a974fafa6480c782
Ingredient from folder: bd0737d46920faabf7794166e20b7883
Ingredient from folder: c277e5ff8eb4b7c4811158018b62ade2
Ingredient from folder: f6200e34bba119053bd2e002328ad029
Ingredient from folder: 5dbabef90c98de08d27b688d48ce476d
Ingredient from folder: 9811ac63a34bf4191cf4ecb913f8c5bc
Ingredient from folder: c4f3a5fc2f9ee2464ecd0a0079ecd51d
Ingredient from folder: 0583a640ee2afc0cd74c88540ba06bad
Ingredient from folder: faa150a49c7e1a6ea9cfc92450fa3102
Ingredient from folder: 0636bccff11f57d695ae431ae6861884
Ingredient from folder: 8e9767b1eaa9aba43da75c0ab4271f6a
Ingredient from folder: 481ce14213c190438e09806b5a64d3b6
Ingredient from folder: 0c731c879dd84ea444c5f2e4364040f6
Ingredient from folder: 8ef43f4bbcde71655fb41193a0f499a6
Ingredient from folder: f2d8f06b0b68aa79edd8dbb68ff3afbc
Ingredient from folder: bd5137c041521b4662f60b601eaf6e2b
Ingredient from folder: 57555a61e05114bd936ef71315dbcd8b
Ingredient from folder: 890605dd9968e9b44de6dc3f4ad7d281
Ingredient from folder: 94dc27f4bfe13209c52cc513ea1e76d2
Ingredient from folder: 118f97d8f4ff5f900c1d56f96a31ee06
Ingredient from folder: dacf9c2dcb4e7d8a88b8cc4fdb2a0b97
Inference at env0_in
Results for at -1


acc_net: 0.6616889804
acc_netm: 0.5744936492
acc_soup: 0.6828012358
acc_swam: 0.6638631422
acc_swa: 0.6797116375
acc_soupswa: 0.6853759011

ece_net: 17.223644468
ece_netm: 25.330150467
ece_soup: 16.170412045
ece_soupswa: 13.885468685
ece_swa: 15.857273781
ece_swam: 17.307143687

net01qstat: 0.9025785081
net01ratio: 0.5585443038
swa0swa1qsta: 0.9844202678
swa0swa1rati: 0.2599653380
swanetqstat: 0.9936417078
swanetratio: 0.1751700680





python3 -m domainbed.scripts.sweep launch --output_dir=/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/ermshhpeoa0317 --command_launcher multi_gpu --datasets OfficeHome --algorithms ERM --single_test_envs --hp swa 1 --hp shared_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home_0316 --test_envs 0

PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66shhpeoa0317 &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed 1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66shhpeoa0317 &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed 2 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66shhpeoa0317 &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=3 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed 3 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66shhpeoa0317 &
wait

HP=EoA CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m domainbed.scripts.sweep launch --output_dir=/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66shhpeoaswa20317 --command_launcher multi_gpu --datasets OfficeHome --algorithms ERM --single_test_envs --hp swa 1 --hp shared_init /gpfswork/rech/edr/utr15kn/dataplace/data/domainbed/inits/home_0316 --test_envs 0 --n_hparams 6 --n_trials 6 --hp swa 2





PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66shhpeoa0317 &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66shhpeoa0317 &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66shhpeoa0317 &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=3 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66shhpeoa0317 &
wait



PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --test_envs 0 --trial_seed -1 --output_dir /data/rame/experiments/domainbed/erm66shhpeoa0317/ --mode ens


PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66shhpeoa0317


PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup ... &
echo 0 &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup ... &
echo 1 &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.soup ... &
echo 2 &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=3 python3 -m domainbed.scripts.soup ... &
echo 3 &
wait



PRETRAINED=0 CUDA_VISIBLE_DEVICES=3 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66shhpeoa0317 --keyacc net --topk 5 &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66shhpeoa0317 --keyacc net --topk 10 &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66shhpeoa0317 --keyacc net --topk 20 &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66shhpeoa0317 --keyacc net --topk 30 &
wait




PRETRAINED=0 CUDA_VISIBLE_DEVICES=3 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66shhpeoa0317 --keyacc swa --topk 5 &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66shhpeoa0317 --keyacc swa --topk 10 &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66shhpeoa0317 --keyacc swa --topk 20 &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66shhpeoa0317 --keyacc swa --topk 30 &


PRETRAINED=0 CUDA_VISIBLE_DEVICES=3 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66shhpeoa0317 --criteriontopk ece_net --topk 5 &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66shhpeoa0317 --criteriontopk ece_net --topk 10 &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66shhpeoa0317 --criteriontopk ece_net --topk 20 &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66shhpeoa0317,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66shhpeoa0317 --criteriontopk ece_net --topk 30 &
wait


PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 2 --cluster algorithm &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 4 --cluster algorithm &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 6 --cluster algorithm &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=3 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 8 --cluster algorithm &


PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 10 &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 20 &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 30 &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=3 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 40 &


SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 10 &
SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 20 &
SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 30 &
SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=3 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 40 &
