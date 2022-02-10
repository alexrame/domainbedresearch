gpu_id=$1
dataset=$2
algo=$3
date=$4

CUDA_VISIBLE_DEVICES=${gpu_id} \
python3 -m domainbed.scripts.sweep launch\
       --datasets $dataset\
       --algorithms ${algo}\
       --data_dir /data/rame/data/domainbed\
       --command_launcher multi_gpu\
       --skip_confirmation\
       --single_test_envs\
       --test_envs $5\
       --output_dir /data/rame/experiments/domainbed/${dataset}_${algo}_${date}/\
       ${*:6:30}
