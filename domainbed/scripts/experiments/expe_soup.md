# commands to test

# PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed 2 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/swaensshhpdeoa0316
# PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --test_envs 0 --trial_seed 2 --output_dir /data/rame/experiments/domainbed/erm66shhpeoa0317/ --mode ens
# SAVE=1 SWAMEMBER=0 PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /data/rame/experiments/domainbed/erm24sheoa0319 --topk 2 --criteriontopk minus_step --cluster dir --trial_seed -1 --regexes net0_net1 --do_ens net
# SAVE=1 SWAMEMBER=0 PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /data/rame/experiments/domainbed/erm24sheoa0319 --topk 2 --criteriontopk step --cluster dir --trial_seed 0 --regexes net0_net1 --do_ens net --mode all
# SAVE=1 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /data/rame/experiments/domainbed/erm24sheoa0319 --topk 12 --criteriontopk minus_step --cluster dir --trial_seed 0 --regexes net0_net1 net0_net2 net0_net3 net0_net4 net0_net5 net0_net6 net0_net7 net0_net8 net0_net9 net0_net10 net0_net11 net0_net12 net0_net13 net0_net14 net0_net15 net0_net16 net0_net17 net0_net18 net0_net19 net0_net20 net0_net21 net0_net22 net0_net23 net1_net2 net1_net3 net1_net4 net1_net5 net1_net6 net1_net7 net1_net8 net1_net9 net1_net10 net1_net11 net1_net12 net1_net13 net1_net14 net1_net15 net1_net16 net1_net17 net1_net18 net1_net19 net1_net20 net1_net21 net1_net22 net1_net23 net2_net3 net2_net4 net2_net5 net2_net6 net2_net7 net2_net8 net2_net9 net2_net10 net2_net11 net2_net12 net2_net13 net2_net14 net2_net15 net2_net16 net2_net17 net2_net18 net2_net19 net2_net20 net2_net21 net2_net22 net2_net23 net3_net4 net3_net5 net3_net6 net3_net7 net3_net8 net3_net9 net3_net10 net3_net11 net3_net12 net3_net13 net3_net14 net3_net15 net3_net16 net3_net17 net3_net18 net3_net19 net3_net20 net3_net21 net3_net22 net3_net23 net4_net5 net4_net6 net4_net7 net4_net8 net4_net9 net4_net10 net4_net11 net4_net12 net4_net13 net4_net14 net4_net15 net4_net16 net4_net17 net4_net18 net4_net19 net4_net20 net4_net21 net4_net22 net4_net23 net5_net6 net5_net7 net5_net8 net5_net9 net5_net10 net5_net11 net5_net12 net5_net13 net5_net14 net5_net15 net5_net16 net5_net17 net5_net18 net5_net19 net5_net20 net5_net21 net5_net22 net5_net23 net6_net7 net6_net8 net6_net9 net6_net10 net6_net11 net6_net12 net6_net13 net6_net14 net6_net15 net6_net16 net6_net17 net6_net18 net6_net19 net6_net20 net6_net21 net6_net22 net6_net23 net7_net8 net7_net9 net7_net10 net7_net11 net7_net12 net7_net13 net7_net14 net7_net15 net7_net16 net7_net17 net7_net18 net7_net19 net7_net20 net7_net21 net7_net22 net7_net23 net8_net9 net8_net10 net8_net11 net8_net12 net8_net13 net8_net14 net8_net15 net8_net16 net8_net17 net8_net18 net8_net19 net8_net20 net8_net21 net8_net22 net8_net23 net9_net10 net9_net11 net9_net12 net9_net13 net9_net14 net9_net15 net9_net16 net9_net17 net9_net18 net9_net19 net9_net20 net9_net21 net9_net22 net9_net23 net10_net11 net10_net12 net10_net13 net10_net14 net10_net15 net10_net16 net10_net17 net10_net18 net10_net19 net10_net20 net10_net21 net10_net22 net10_net23 net11_net12 net11_net13 net11_net14 net11_net15 net11_net16 net11_net17 net11_net18 net11_net19 net11_net20 net11_net21 net11_net22 net11_net23 net12_net13 net12_net14 net12_net15 net12_net16 net12_net17 net12_net18 net12_net19 net12_net20 net12_net21 net12_net22 net12_net23 net13_net14 net13_net15 net13_net16 net13_net17 net13_net18 net13_net19 net13_net20 net13_net21 net13_net22 net13_net23 net14_net15 net14_net16 net14_net17 net14_net18 net14_net19 net14_net20 net14_net21 net14_net22 net14_net23 net15_net16 net15_net17 net15_net18 net15_net19 net15_net20 net15_net21 net15_net22 net15_net23 net16_net17 net16_net18 net16_net19 net16_net20 net16_net21 net16_net22 net16_net23 net17_net18 net17_net19 net17_net20 net17_net21 net17_net22 net17_net23 net18_net19 net18_net20 net18_net21 net18_net22 net18_net23 net19_net20 net19_net21 net19_net22 net19_net23 net20_net21 net20_net22 net20_net23 net21_net22 net21_net23 net22_net23 --do_ens 1


HESSIAN=0 SAVE=0 STEPS=mod100 PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /data/rame/experiments/domainbed/erm24sheoa0319/ef77dcccb229850e902de8f4a8bd47a4 --topk 30 --mode all


DEBUG=1 HESSIAN=1 SAVE=0 STEPS=mod10 PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /data/rame/experiments/domainbed/erm24sheoa0319/ef77dcccb229850e902de8f4a8bd47a4 --topk 30 --mode iter_1_16_10 &


STEPS="mod100"

HESSIAN=1 SAVE=0 STEPS=mod100 PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/ae1b6d92c1673c382dff447c31ac556a --topk 30 --mode iter_1_16_10 &
HESSIAN=1 SAVE=0 STEPS=mod100 PRETRAINED=0 CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/ae1b6d92c1673c382dff447c31ac556a --topk 30 --mode iter_16_30_10 &
HESSIAN=1 SAVE=0 PRETRAINED=0 CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm320sh0319 --topk 30 --mode iter_1_15_10 &
HESSIAN=1 SAVE=0 PRETRAINED=0 CUDA_VISIBLE_DEVICES=3 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm320sh0319 --topk 30 --mode iter_16_30_10 &

# HESSIAN=1 SCORES=5000_3000 SAVE=1 SWAMEMBER=0 PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319 --topk 2 --criteriontopk step --cluster dir --trial_seed -1 --regexes net0_net1 --do_ens net --mode all
# SCORES=5000_4000_3000_2000_1000 SAVE=1 SWAMEMBER=0 PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319 --topk 5 --criteriontopk step --cluster dir --trial_seed 0 --regexes net0_net1 --do_ens net --mode all

# HESSIAN=1 SCORES=5000_3000 SAVE=1 SWAMEMBER=0 PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm320sh0319 --topk 30 --trial_seed 0 --regexes net0_net1 --do_ens net --mode all &
# HESSIAN=1 SCORES=5000_3000 SAVE=1 SWAMEMBER=0 PRETRAINED=0 CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm320sh0319 --topk 30 --trial_seed 1 --regexes net0_net1 --do_ens net --mode all &
# HESSIAN=1 SCORES=5000_3000 SAVE=1 SWAMEMBER=0 PRETRAINED=0 CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm320sh0319 --topk 30 --trial_seed 2 --regexes net0_net1 --do_ens net --mode all &
# HESSIAN=1 SCORES=5000_3000 SAVE=1 SWAMEMBER=0 PRETRAINED=0 CUDA_VISIBLE_DEVICES=3 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm320sh0319 --topk 60 --trial_seed -1 --regexes net0_net1 --do_ens net --mode all &
# wait

# HESSIAN=1 SCORES=3000 SAVE=1 SWAMEMBER=0 PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm320sh0319 --topk 60 --trial_seed -1 --regexes net0_net1 --do_ens net --mode all &
# HESSIAN=1 SCORES=5000 SAVE=1 SWAMEMBER=0 PRETRAINED=0 CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm320sh0319 --topk 60 --trial_seed -1 --regexes net0_net1 --do_ens net --mode all &
# SAVE=1 SWAMEMBER=0 PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /data/rame/experiments/domainbed/erm24sheoa0319 --topk 10 --trial_seed -1 --regexes net0_net1 --do_ens net --mode all



# clean
BS=12 PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --test_envs 0 --trial_seed -1 --output_dir /data/rame/experiments/domainbed/erm66shhpeoa0317/ --cluster trial_seed --topk 1 --criteriontopk acc_net
CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.train --algorithm Ensembling --dataset ColoredMNIST --test_env 2 --hp swa 1

HESSIAN=100 PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --test_envs 0 --trial_seed -1 --output_dir /data/rame/experiments/domainbed/erm66shhpeoa0317/ --topk 1 --criteriontopk acc_net --selection oracle --holdout_fraction 0.95

#

PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66sh0319 --topk 25 &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66sh0319,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318 --topk 20 &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66sh0319,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318 --topk 30 &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66sh0319,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318 --topk 40 &
wait



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



HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318 --criteriontopk acc_net --topk 20 &
HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318 --criteriontopk acc_net --topk 20 &
HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318 --criteriontopk acc_net --topk 20 &
HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=3 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 20 &
wait

HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 20 &
HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 20 &
HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 20 &
HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=3 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318 --criteriontopk acc_net --topk 20 &


/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 120 &
HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=3 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 20 &

/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 120 &



{'Accuracies/acc_net': 0.6440049443757726, 'Calibration/ece_net': 18.87351588753206, 'Accuracies/acc_swa': 0.6744952616398846, 'Calibration/ece_swa': 16.417679756720176, 'Accuracies/acc_netts': 0.6440049443757726, 'Calibration/ece_netts': 17.19502860723968, 'Accuracies/acc_swats': 0.6440049443757726, 'Calibration/ece_swats': 17.17603193587563, 'Accuracies/acc_soup': 0.6650185414091471, 'Calibration/ece_soup': 18.320240126404684, 'Accuracies/acc_soupswa': 0.6847960444993819, 'Calibration/ece_soupswa': 14.45842759386718, 'Accuracies/acc_netm': 0.5762257931602802, 'Calibration/ece_netm': 25.650187453456905, 'Accuracies/acc_swam': 0.6601428375223184, 'Calibration/ece_swam': 17.38016808188325, 'Diversity/swa0swa1ratio': 0.3991228070175438, 'Diversity/swa0swa1qstat': 0.9640160874771219, 'Diversity/swa0swa1CKAC': 0.12154906988143921, 'Diversity/net01ratio': 0.6360103626943004, 'Diversity/net01qstat': 0.8742944864612316, 'Diversity/net01CKAC': 0.2517595887184143}


{'Accuracies/acc_net': 0.6440049443757726, 'Calibration/ece_net': 18.87351588753206, 'Accuracies/acc_swa': 0.6744952616398846, 'Calibration/ece_swa': 16.417679756720176, 'Accuracies/acc_netts': 0.6440049443757726, 'Calibration/ece_netts': 17.19502860723968, 'Accuracies/acc_swats': 0.6440049443757726, 'Calibration/ece_swats': 17.17603193587563, 'Accuracies/acc_soup': 0.6650185414091471, 'Calibration/ece_soup': 18.320240126404684, 'Accuracies/acc_soupswa': 0.6847960444993819, 'Calibration/ece_soupswa': 14.45842759386718, 'Accuracies/acc_netm': 0.5762257931602802, 'Calibration/ece_netm': 25.650187453456905, 'Accuracies/acc_swam': 0.6601428375223184, 'Calibration/ece_swam': 17.38016808188325, 'Diversity/swa0swa1ratio': 0.3991228070175438, 'Diversity/swa0swa1qstat': 0.9640160874771219, 'Diversity/swa0swa1CKAC': 0.12154906988143921, 'Diversity/net01ratio': 0.6360103626943004, 'Diversity/net01qstat': 0.8742944864612316, 'Diversity/net01CKAC': 0.2517595887184143}


{'Accuracies/acc_net': 0.6440049443757726, 'Calibration/ece_net': 18.87351588753206, 'Accuracies/acc_swa': 0.6744952616398846, 'Calibration/ece_swa': 16.417679756720176, 'Accuracies/acc_netts': 0.6440049443757726, 'Calibration/ece_netts': 17.19502860723968, 'Accuracies/acc_swats': 0.6440049443757726, 'Calibration/ece_swats': 17.17603193587563, 'Accuracies/acc_soup': 0.6650185414091471, 'Calibration/ece_soup': 18.319096185687545, 'Accuracies/acc_soupswa': 0.6847960444993819, 'Calibration/ece_soupswa': 14.459289133941681, 'Accuracies/acc_netm': 0.5762257931602802, 'Calibration/ece_netm': 25.650187453456905, 'Accuracies/acc_swam': 0.6601428375223184, 'Calibration/ece_swam': 17.38016808188325, 'Diversity/swa0swa1ratio': 0.3991228070175438, 'Diversity/swa0swa1qstat': 0.9640160874771219, 'Diversity/swa0swa1CKAC': 0.12154906988143921, 'Diversity/net01ratio': 0.6360103626943004, 'Diversity/net01qstat': 0.8742944864612316, 'Diversity/net01CKAC': 0.2517595887184143}




HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 20 &
HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318 --criteriontopk acc_net --topk 20 &
HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318 --criteriontopk acc_net --topk 20 &
HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=3 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 20 &
wait



HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 4 --cluster algorithm &
HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 6 --cluster algorithm &
HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 8 --cluster algorithm &
HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=3 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 1 --cluster trial_seed algorithm &
wait

HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318 --criteriontopk acc_net --topk 20 &
HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318 --criteriontopk acc_net --topk 20 &
HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=3 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 20 &
wait




HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --topk 2 --cluster algorithm --regexes net0_net1 net0_net2 net0_net3 net0_net4 net0_net5 net0_net6 net0_net7 net0_net8 net0_net9 net1_net2 net1_net3 net1_net4 net1_net5 net1_net6 net1_net7 net1_net8 net1_net9 net2_net3 net2_net4 net2_net5 net2_net6 net2_net7 net2_net8 net2_net9 net3_net4 net3_net5 net3_net6 net3_net7 net3_net8 net3_net9 net4_net5 net4_net6 net4_net7 net4_net8 net4_net9 net5_net6 net5_net7 net5_net8 net5_net9 net6_net7 net6_net8 net6_net9 net7_net8 net7_net9 net8_net9 --do_ens 1 &
HESSIAN=0 SWAMEMBER=0 PRETRAINED=0 CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --topk 2 --cluster algorithm --regexes swa0_swa1 swa0_swa2 swa0_swa3 swa0_swa4 swa0_swa5 swa0_swa6 swa0_swa7 swa0_swa8 swa0_swa9 swa1_swa2 swa1_swa3 swa1_swa4 swa1_swa5 swa1_swa6 swa1_swa7 swa1_swa8 swa1_swa9 swa2_swa3 swa2_swa4 swa2_swa5 swa2_swa6 swa2_swa7 swa2_swa8 swa2_swa9 swa3_swa4 swa3_swa5 swa3_swa6 swa3_swa7 swa3_swa8 swa3_swa9 swa4_swa5 swa4_swa6 swa4_swa7 swa4_swa8 swa4_swa9 swa5_swa6 swa5_swa7 swa5_swa8 swa5_swa9 swa6_swa7 swa6_swa8 swa6_swa9 swa7_swa8 swa7_swa9 swa8_swa9 --do_ens 1 &
HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --topk 2 --cluster algorithm --regexes swa0_swa1 swa0_swa2 swa0_swa3 swa0_swa4 swa0_swa5 swa0_swa6 swa0_swa7 swa0_swa8 swa0_swa9 swa1_swa2 swa1_swa3 swa1_swa4 swa1_swa5 swa1_swa6 swa1_swa7 swa1_swa8 swa1_swa9 swa2_swa3 swa2_swa4 swa2_swa5 swa2_swa6 swa2_swa7 swa2_swa8 swa2_swa9 swa3_swa4 swa3_swa5 swa3_swa6 swa3_swa7 swa3_swa8 swa3_swa9 swa4_swa5 swa4_swa6 swa4_swa7 swa4_swa8 swa4_swa9 swa5_swa6 swa5_swa7 swa5_swa8 swa5_swa9 swa6_swa7 swa6_swa8 swa6_swa9 swa7_swa8 swa7_swa9 swa8_swa9 --do_ens 1 &
wait


SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318 --criteriontopk acc_net --topk 20 &

SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318 --criteriontopk acc_net --topk 20 &
SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318 --criteriontopk acc_net --topk 20 &

HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 25 &
HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 30 &
HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 40 &
HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=3 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 50 &
wait
--regexes swa0_swa1 swa0_swa2 swa0_swa3 swa0_swa4 swa0_swa5 swa1_swa2 swa1_swa3 swa1_swa4 swa1_swa5 swa2_swa3 swa2_swa4 swa2_swa5 swa3_swa4 swa3_swa5 swa4_swa5 --do_ens 1 &




SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66sh0319 --criteriontopk acc_net --topk 20 &
SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 1 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66sh0319 --criteriontopk acc_net --topk 20 &
SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 2 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66sh0319 --criteriontopk acc_net --topk 20 &
SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=3 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 3 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66shhpeoaswa20317 --criteriontopk acc_net --topk 20 &
wait



SWAMEMBER=0 PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66shhpeoaswa20317 --criteriontopk acc_net --topk 20 --do_ens 1 &
SWAMEMBER=0 PRETRAINED=0 CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 1 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66shhpeoaswa20317 --criteriontopk acc_net --topk 20 --do_ens 1 &
SWAMEMBER=0 PRETRAINED=0 CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 2 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66shhpeoaswa20317 --criteriontopk acc_net --topk 20 --do_ens 1 &
SWAMEMBER=0 PRETRAINED=0 CUDA_VISIBLE_DEVICES=3 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 3 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66shhpeoaswa20317 --criteriontopk acc_net --topk 20 --do_ens 1 &
wait



Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/ae1b6d92c1673c382dff447c31ac556a/5000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/ae1b6d92c1673c382dff447c31ac556a/4800
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/ae1b6d92c1673c382dff447c31ac556a/5000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/ae1b6d92c1673c382dff447c31ac556a/4500
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/ae1b6d92c1673c382dff447c31ac556a/5000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/ae1b6d92c1673c382dff447c31ac556a/4000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/ae1b6d92c1673c382dff447c31ac556a/5000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/ae1b6d92c1673c382dff447c31ac556a/3000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/ae1b6d92c1673c382dff447c31ac556a/4800
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/ae1b6d92c1673c382dff447c31ac556a/4500
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/ae1b6d92c1673c382dff447c31ac556a/4800
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/ae1b6d92c1673c382dff447c31ac556a/4000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/ae1b6d92c1673c382dff447c31ac556a/4800
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/ae1b6d92c1673c382dff447c31ac556a/3000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/ae1b6d92c1673c382dff447c31ac556a/4500
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/ae1b6d92c1673c382dff447c31ac556a/4000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/ae1b6d92c1673c382dff447c31ac556a/4500
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/ae1b6d92c1673c382dff447c31ac556a/3000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/ae1b6d92c1673c382dff447c31ac556a/4000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/ae1b6d92c1673c382dff447c31ac556a/3000



l.append({"net":0.606, "netm":0.572, "soup":0.615, "soupswa" :0.666, "df" :0.127, "dr":0.508, "hess": 19077.772, "d": 200})
l.append({"net":0.612, "netm":0.578, "soup":0.619, "soupswa" :0.665, "df" :0.153, "dr":0.476, "hess": 17534.173, "d": 500})
l.append({"net":0.622, "netm":0.589, "soup":0.635, "soupswa" :0.663, "df" :0.141, "dr":0.466, "hess": 18118.590, "d": 1000})
l.append({"net":0.624, "netm":0.590, "soup":0.640, "soupswa" :0.661, "df" :0.154, "dr":0.584, "hess": 17994.414, "d": 2000})
l.append({"net":0.603, "netm":0.577, "soup":0.613, "soupswa" :0.664, "df" :0.144, "dr":0.482, "hess": 25000.919, "d": 300})
l.append({"net":0.614, "netm":0.588, "soup":0.621, "soupswa" :0.663, "df" :0.117, "dr":0.494, "hess": 21223.235, "d": 800})
l.append({"net":0.615, "netm":0.589, "soup":0.625, "soupswa" :0.660, "df" :0.182, "dr":0.544, "hess": 20945.467, "d": 1800})
l.append({"net":0.616, "netm":0.594, "soup":0.626, "soupswa" :0.661, "df" :0.169, "dr":0.434, "hess": 22047.766, "d": 500})
l.append({"net":0.619, "netm":0.595, "soup":0.628, "soupswa" :0.661, "df" :0.161, "dr":0.521, "hess": 19676.541, "d": 1500})
l.append({"net":0.630, "netm":0.606, "soup":0.641, "soupswa" :0.660, "df" :0.165, "dr":0.481, "hess": 22340.328, "d": 1000})



Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/a940d651058fe4b036ed2734590224d9/5000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/a940d651058fe4b036ed2734590224d9/3000
l.append({"net":0.629, "netm":0.600, "soup":0.642, "soupswa":0.663, "df":0.169, "dr":0.521, "hess": 17970.371, "same": 1})
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/a940d651058fe4b036ed2734590224d9/5000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/870fe24b8be0b86a0503d229fd1752ad/5000
l.append({"net":0.620, "netm":0.591, "soup":0.642, "soupswa":0.672, "df":0.253, "dr":0.574, "hess": 15387.298, "same": 0})
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/a940d651058fe4b036ed2734590224d9/5000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/870fe24b8be0b86a0503d229fd1752ad/3000
l.append({"net":0.628, "netm":0.588, "soup":0.643, "soupswa":0.670, "df":0.294, "dr":0.557, "hess": 16450.27, "same": 0})
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/a940d651058fe4b036ed2734590224d9/5000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/d2ee7f8ca16d2f43b0962619641c8c64/5000
l.append({"net":0.632, "netm":0.591, "soup":0.649, "soupswa":0.672, "df":0.399, "dr":0.579, "hess": 14323.597, "same": 0})
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/a940d651058fe4b036ed2734590224d9/5000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/d2ee7f8ca16d2f43b0962619641c8c64/3000
l.append({"net":0.617, "netm":0.588, "soup":0.635, "soupswa":0.670, "df":0.358, "dr":0.582, "hess": 15570.857, "same": 0})
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/a940d651058fe4b036ed2734590224d9/5000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/2f1c184db1532874464b694a1142dd52/5000
l.append({"net":0.632, "netm":0.592, "soup":0.648, "soupswa":0.670, "df":0.262, "dr":0.540, "hess": 13460.164})
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/a940d651058fe4b036ed2734590224d9/5000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/2f1c184db1532874464b694a1142dd52/3000
l.append({"net":0.610, "netm":0.577, "soup":0.637, "soupswa":0.670, "df":0.286, "dr":0.614, "hess": 14454.661})
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/a940d651058fe4b036ed2734590224d9/5000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/ef77dcccb229850e902de8f4a8bd47a4/5000
l.append({"net":0.623, "netm":0.583, "soup":0.646, "soupswa":0.674, "df":0.240, "dr":0.589, "hess": 14110.631})
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/a940d651058fe4b036ed2734590224d9/5000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/ef77dcccb229850e902de8f4a8bd47a4/3000
l.append({"net":0.623, "netm":0.580, "soup":0.647, "soupswa":0.670, "df":0.175, "dr":0.564, "hess": 15217.058})
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/a940d651058fe4b036ed2734590224d9/5000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/89eaeae298e48ac4e987a494dab1fd05/5000
l.append({"net":0.633, "netm":0.590, "soup":0.651, "soupswa":0.677, "df":0.317, "dr":0.694, "hess": 14078.031})
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/a940d651058fe4b036ed2734590224d9/5000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/89eaeae298e48ac4e987a494dab1fd05/3000
l.append({"net":0.627, "netm":0.590, "soup":0.647, "soupswa":0.676, "df":0.187, "dr":0.594, "hess": 14147.583})
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/a940d651058fe4b036ed2734590224d9/5000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/ae1b6d92c1673c382dff447c31ac556a/5000
l.append({"net":0.619, "netm":0.584, "soup":0.643, "soupswa":0.673, "df":0.210, "dr":0.604, "hess": 13956.838})
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/a940d651058fe4b036ed2734590224d9/5000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/ae1b6d92c1673c382dff447c31ac556a/3000
l.append({"net":0.637, "netm":0.601, "soup":0.654, "soupswa":0.670, "df":0.263, "dr":0.613, "hess": 15227.965})
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/a940d651058fe4b036ed2734590224d9/5000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/339f6d0ca850f72a0494bac944730d42/5000
l.append({"net":0.617, "netm":0.583, "soup":0.640, "soupswa":0.678, "df":0.233, "dr":0.576, "hess": 13417.053})
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/a940d651058fe4b036ed2734590224d9/5000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/339f6d0ca850f72a0494bac944730d42/3000
l.append({"net":0.635, "netm":0.599, "soup":0.656, "soupswa":0.675, "df":0.209, "dr":0.643, "hess": 14790.377})
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/a940d651058fe4b036ed2734590224d9/3000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/870fe24b8be0b86a0503d229fd1752ad/5000
l.append({"net":0.629, "netm":0.596, "soup":0.646, "soupswa":0.672, "df":0.216, "dr":0.576, "hess": 16969.272})
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/a940d651058fe4b036ed2734590224d9/3000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/870fe24b8be0b86a0503d229fd1752ad/3000
l.append({"net":0.628, "netm":0.593, "soup":0.647, "soupswa":0.669, "df":0.216, "dr":0.533, "hess": 18696.943})
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/a940d651058fe4b036ed2734590224d9/3000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/d2ee7f8ca16d2f43b0962619641c8c64/5000
l.append({"net":0.632, "netm":0.596, "soup":0.654, "soupswa":0.672, "df":0.351, "dr":0.575, "hess": 14708.428})
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/a940d651058fe4b036ed2734590224d9/3000
Ingredient from folder: /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm24sheoa0319/d2ee7f8ca16d2f43b0962619641c8c64/3000
l.append({"net":0.624, "netm":0.593, "soup":0.644, "soupswa":0.668, "df":0.306, "dr":0.612, "hess": 15324.569})

l35.append({"net":0.628, "netm":0.588, "soup":0.643, "soupswa":0.670, "df":0.294, "dr":0.557, "hess": 16450.277})
l35.append({"net":0.617, "netm":0.588, "soup":0.635, "soupswa":0.670, "df":0.358, "dr":0.582, "hess": 15570.857})
l35.append({"net":0.610, "netm":0.577, "soup":0.637, "soupswa":0.670, "df":0.286, "dr":0.614, "hess": 14454.661})
l35.append({"net":0.623, "netm":0.580, "soup":0.647, "soupswa":0.670, "df":0.175, "dr":0.564, "hess": 15217.058})
l35.append({"net":0.627, "netm":0.590, "soup":0.647, "soupswa":0.676, "df":0.187, "dr":0.594, "hess": 14147.583})
l35.append({"net":0.637, "netm":0.601, "soup":0.654, "soupswa":0.670, "df":0.263, "dr":0.613, "hess": 15227.965})
l35.append({"net":0.635, "netm":0.599, "soup":0.656, "soupswa":0.675, "df":0.209, "dr":0.643, "hess": 14790.377})
l35.append({"net":0.629, "netm":0.596, "soup":0.646, "soupswa":0.672, "df":0.216, "dr":0.576, "hess": 16969.272})
l35.append({"net":0.632, "netm":0.596, "soup":0.654, "soupswa":0.672, "df":0.351, "dr":0.575, "hess": 14708.428})



HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318 --criteriontopk acc_net --topk 25 &
HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 30 &
HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 40 &
HESSIAN=0 SWAMEMBER=4 PRETRAINED=0 CUDA_VISIBLE_DEVICES=3 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --criteriontopk acc_net --topk 50 &



PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66sh0319,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/fishr66swa5sheoa0318 --topk 40 &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66sh0319,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/coral66swa5sheoa0318 --topk 40 &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66sh0319,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/mixup66swa5sheoa0318 --topk 40 &
PRETRAINED=0 CUDA_VISIBLE_DEVICES=3 python3 -m domainbed.scripts.soup --algorithm Soup --dataset OfficeHome --mode ens --test_envs 0 --trial_seed -1 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66sh0319,/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/gdro66swa5sheoa0318 --topk 40 &
wait

len(accs): 6 for test_env: 0 and trial_seed: 4 for dataset: OfficeHome
len(accs): 6 for test_env: 0 and trial_seed: 0 for dataset: OfficeHome
len(accs): 6 for test_env: 0 and trial_seed: 1 for dataset: OfficeHome
len(accs): 6 for test_env: 0 and trial_seed: 5 for dataset: OfficeHome
len(accs): 6 for test_env: 0 and trial_seed: 2 for dataset: OfficeHome
len(accs): 6 for test_env: 0 and trial_seed: 3 for dataset: OfficeHome



\subsection{Model selection: training-domain validation set}
len(accs): 6 for test_env: 2 and trial_seed: 0 for dataset: OfficeHome
len(accs): 6 for test_env: 2 and trial_seed: 4 for dataset: OfficeHome
len(accs): 6 for test_env: 2 and trial_seed: 2 for dataset: OfficeHome
len(accs): 6 for test_env: 3 and trial_seed: 3 for dataset: OfficeHome
len(accs): 6 for test_env: 3 and trial_seed: 4 for dataset: OfficeHome
len(accs): 6 for test_env: 1 and trial_seed: 4 for dataset: OfficeHome
len(accs): 6 for test_env: 3 and trial_seed: 0 for dataset: OfficeHome
len(accs): 6 for test_env: 1 and trial_seed: 0 for dataset: OfficeHome
len(accs): 6 for test_env: 1 and trial_seed: 2 for dataset: OfficeHome
len(accs): 6 for test_env: 1 and trial_seed: 3 for dataset: OfficeHome
len(accs): 6 for test_env: 2 and trial_seed: 1 for dataset: OfficeHome
len(accs): 6 for test_env: 3 and trial_seed: 2 for dataset: OfficeHome
len(accs): 6 for test_env: 1 and trial_seed: 5 for dataset: OfficeHome
len(accs): 6 for test_env: 3 and trial_seed: 5 for dataset: OfficeHome
len(accs): 6 for test_env: 2 and trial_seed: 3 for dataset: OfficeHome
len(accs): 6 for test_env: 2 and trial_seed: 5 for dataset: OfficeHome
len(accs): 6 for test_env: 3 and trial_seed: 1 for dataset: OfficeHome
len(accs): 6 for test_env: 1 and trial_seed: 1 for dataset: OfficeHome


/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66sh0319/b14e9d652a6afb0dc99eb7d3f903fbc5
/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66sh0319/9f575d673abcd472add57aecbf773e98
/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66sh0319/26a5d2ee460f68e77b1f6bce95f1ecc9
/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66sh0319/76c8e21c393a6a6ecf4ede0324213b30
/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66sh0319/f9112dbfe3ec0778e2bb88b1031c99c5
/gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm66sh0319/860d143231d85730b21622125130f2d9


HESSIAN=1 STEPS=5000_3000 SAVE=1 SWAMEMBER=0 PRETRAINED=0 CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm320sh0319 --topk 30 --trial_seed 0,1,2,3,4 --regexes net0_net1 --do_ens net --mode all &
HESSIAN=1 STEPS=5000_3000 SAVE=1 SWAMEMBER=0 PRETRAINED=0 CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm320sh0319 --topk 30 --trial_seed 5,6,7,8,9 --regexes net0_net1 --do_ens net --mode all &
HESSIAN=1 STEPS=5000_3000 SAVE=1 SWAMEMBER=0 PRETRAINED=0 CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm320sh0319 --topk 30 --trial_seed 10,11,12,13,14 --regexes net0_net1 --do_ens net --mode all &
HESSIAN=1 STEPS=5000_3000 SAVE=1 SWAMEMBER=0 PRETRAINED=0 CUDA_VISIBLE_DEVICES=3 python3 -m domainbed.scripts.soup --dataset OfficeHome --test_envs 0 --output_dir /gpfswork/rech/edr/utr15kn/dataplace/experiments/domainbed/erm320sh0319 --topk 60 --trial_seed 15,16,17,18,19 --regexes net0_net1 --do_ens net --mode all &
wait
