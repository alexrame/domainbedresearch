gpu_id=$1
dataset=$2
test_env=$3
algo=$4

CUDA_VISIBLE_DEVICES=${gpu_id} \
python3 -m domainbed.scripts.train\
       --datasets $dataset\
       --algorithms ${algo}\
       --data_dir /data/rame/data/domainbed\
       --test_env ${test_env}\
       ${*:5:10}

