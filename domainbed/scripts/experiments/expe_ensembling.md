CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.train --test_envs 2 --algorithm Ensembling --dataset ColoredMNISTClean --data_dir /data/rame/data/domainbed


CUDA_VISIBLE_DEVICES=2 python3 -m domainbed.scripts.train --test_envs 2 --algorithm Ensembling --dataset ColoredMNISTClean --data_dir /data/rame/data/domainbed --hp diversity_loss FeaturesDiversity --hp lambda_diversity_loss 1.0 --hp conditional_d 1
a58d8b9e3ad944ff9870cb04edaa205a

757f5dc342a042e5879ac65498c07f39

export LOGDIR=no:/data/rame/mlruns/62/a58d8b9e3ad944ff9870cb04edaa205a,ceb01:/data/rame/mlruns/62/da382eba4c52407585cf271bd51736ab,ceb1:/data/rame/mlruns/62/380e20a1eaa64f36889f0aaff7df9072,dice1:/data/rame/mlruns/62/f2e7b587e05d460c8bd63bfb866e2f78,dicenew:/data/rame/mlruns/62/8fd18051a1a84303b1a68f2a55de354f


9aed42cf43fe4e438f5e946da7ab25ee
da796755302e4ff5afb3fb87ce09f953
b37e4a739afb469f9f8e1ce29ad53b5d
1edb7f4fa8cd42dc9241a0ec30a3d1d1

export LOGDIR=e01:${ML}/62/1edb7f4fa8cd42dc9241a0ec30a3d1d1,0:${ML}/62/b37e4a739afb469f9f8e1ce29ad53b5d,e1:${ML}/62/da796755302e4ff5afb3fb87ce09f953,1:${ML}/62/9aed42cf43fe4e438f5e946da7ab25ee



CUDA_VISIBLE_DEVICES=3 python3 -m domainbed.scripts.train --test_envs 2 --algorithm Ensembling --dataset ColoredMNISTClean --data_dir /data/rame/data/domainbed --hp diversity_loss FeaturesDiversity --hp lambda_diversity_loss 0.01 --hp conditional_d 1 --hp penalty_anneal_iters 1500


CUDA_VISIBLE_DEVICES=3 python3 -m domainbed.scripts.train --test_envs 2 --algorithm Ensembling --dataset ColoredMNISTClean --data_dir /data/rame/data/domainbed --hp lambda_diversity_loss 0.01
--hp num_hidden_layers 2 --hp hidden_size 10


Run ID,Name,Source Type,Source Name,User,Status,lambda_diversity_loss
export LOGDIR=1:${ML}/63/00bdef27a82f49249e88a882a04fcd78/artifacts,01:${ML}/63/fab972cee78d46c096b54f2e21aad85d/artifacts,2:${ML}/63/d78d26947ac24d649c41a08d18d845d4/artifacts,0:${ML}/63/91dd7f90d0e142bd961c62ecbac5c2b8/artifacts


Run ID,Name,Source Type,Source Name,User,Status,lambda_diversity_loss,lambda_ib_firstorder

export LOGDIR=06:${ML}/63/775697e722c14ca681a9fa781e834a72,08:${ML}/63/8c78aacb2cd7441a95add9291adb0a8b,04:${ML}/63/84df6da53a3e42229fe15bba84b18c3f,02:${ML}/63/8f17682d036643a19fc3468c1b21f70c,1:${ML}/63/00bdef27a82f49249e88a882a04fcd78,01:${ML}/63/fab972cee78d46c096b54f2e21aad85d,2:${ML}/63/d78d26947ac24d649c41a08d18d845d4,0:${ML}/63/91dd7f90d0e142bd961c62ecbac5c2b8

export LOGDIR=01:${ML}/64/2cd6741938d54bb48333e2a8ea3982c9/artifacts,0:${ML}/64/581f57d7bb4e4b74a7d9845fb8a2904e/artifacts,1:${ML}/64/be58506107eb4e48a4aed136c4113e2b/artifacts,05:${ML}/64/3c4c61a0c2724ef09cd976ebd73dd86c/artifacts


export LOGDIR=01_0.1:${ML}/63/4a51370578d24a3aa7bb2630ff66438f/artifacts,01_0.02:${ML}/63/cd9d5b41055b4c9c8a334c787a952015/artifacts,01_0.01:${ML}/63/2db59b948fc84645ac42eb1d64390e85/artifacts,06_0:${ML}/63/775697e722c14ca681a9fa781e834a72/artifacts,08_0:${ML}/63/8c78aacb2cd7441a95add9291adb0a8b/artifacts,04_0:${ML}/63/84df6da53a3e42229fe15bba84b18c3f/artifacts,02_0:${ML}/63/8f17682d036643a19fc3468c1b21f70c/artifacts,1_0:${ML}/63/00bdef27a82f49249e88a882a04fcd78/artifacts,01_0:${ML}/63/fab972cee78d46c096b54f2e21aad85d/artifacts,2_0:${ML}/63/d78d26947ac24d649c41a08d18d845d4/artifacts,0_0:${ML}/63/91dd7f90d0e142bd961c62ecbac5c2b8/artifacts

export LOGDIR=0:${ML}/63/8ec09e6600ab4ccaba8ebcee3f7b1283/artifacts,01:${ML}/63/220a4847537f46e6b0347f819444be32/artifacts,1:${ML}/63/f3737ff9ee5947adaf8ef178a5176146/artifacts


export LOGDIR=00:${ML}/63/c489d4037bf34118a14f375263a7ac1c/artifacts,01:${ML}/63/3632a4a66a2548dcbf5f5def5745bd4d/artifacts,10:${ML}/63/1fb27a2fca5847d2a51b670a4b94e3b9/artifacts
