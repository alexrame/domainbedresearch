
# test 12

## erm

CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train --data_dir=./domainbed/data/ECML/ --algorithm ERM --dataset ChallengeECML --train_envs 0 1  --test_envs 12 --hparams '{"mlp_width":40, "mlp_dropout":0.0, "mlp_depth":3}'

env0_in_acc   env0_out_acc  env1_in_acc   env1_out_acc  env2_in_acc   env2_out_acc  epoch         loss          mem_gb        step          step_time
0.6236212500  0.6233050000  0.6121687500  0.6112200000  0.7626284654  0.7612238080  0.3711866373  0.6348411021  0.0001916885  5000          0.0050101638

## swa 100
0.6162250000  0.6160950000  0.6018112500  0.6009300000  0.7579375942  0.7564076391  0.3711866373  0.6348411021  0.0002298355  5000          0.0054695344


# all tests


## erm
PATH_ECML=/data/rame/data/ECML/ CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train --data_dir=/data/rame/data/ECML/ --algorithm ERM --dataset ChallengeECML --train_envs 0 1  --test_envs 2 3 4 5 6 7 8 9 10 11 12 --hparams '{"mlp_width":40, "mlp_dropout":0.0, "mlp_depth":3}'

0.6223975000  0.6220200000  0.5649015925  0.5664753213  0.7053101900  0.7043703278  0.7573692147  0.7590059576  0.6110575000  0.6100300000  0.6615612500  0.6617300000  0.6723005374  0.6733544219  0.4822520881  0.4851429376  0.5325722372  0.5308525507  0.5746622356  0.5785171197  0.6770180089  0.6774262252  0.6214084677  0.6191049569  0.6211669612  0.6227703733  1.4082151753  0.6902092707  0.0001554489  5000          0.0034339106


# swa

PATH_ECML=/data/rame/data/ECML/ CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train --data_dir=/data/rame/data/ECML/ --algorithm SWA --dataset ChallengeECML --train_envs 0 1  --test_envs 2 3 4 5 6 7 8 9 10 11 12 --hparams '{"mlp_width":40, "mlp_dropout":0.0, "mlp_depth":3, "swa_start_iter": 100}'


env0_in_acc   env0_out_acc  env10_in_acc  env10_out_ac  env11_in_acc  env11_out_ac  env12_in_acc  env12_out_ac  env1_in_acc   env1_out_acc  env2_in_acc   env2_out_acc  env3_in_acc   env3_out_acc  env4_in_acc   env4_out_acc  env5_in_acc   env5_out_acc env6_in_acc   env6_out_acc  env7_in_acc   env7_out_acc  env8_in_acc   env8_out_acc  env9_in_acc   env9_out_acc  epoch         loss          mem_gb        step          step_time
0.6223975000  0.6220200000  0.5649015925  0.5664753213  0.7053101900  0.7043703278  0.7573692147  0.7590059576  0.6110575000  0.6100300000  0.6615612500  0.6617300000  0.6723005374  0.6733544219  0.4822520881  0.4851429376  0.5325722372  0.5308525507  0.5746622356  0.5785171197  0.6770180089  0.6774262252  0.6214084677  0.6191049569  0.6211669612  0.6227703733  1.4082151753  0.6902092707  0.0001554489  5000          0.0037787390

## swa1000
PATH_ECML=/data/rame/data/ECML/ CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.train --data_dir=/data/rame/data/ECML/ --algorithm SWA --dataset ChallengeECML --train_envs 0 1  --test_envs 2 3 4 5 6 7 8 9 10 11 12 --hparams '{"mlp_width":40, "mlp_dropout":0.0, "mlp_depth":3, "swa_start_iter": 1000}'


0.6220662500  0.6218650000  0.5633481898  0.5646791830  0.7043441258  0.7034847614  0.7573135367  0.7589038808  0.6110837500  0.6105800000  0.6610887500  0.6612100000  0.6717015355  0.6728361825  0.4820848626  0.4849316998  0.5321442298  0.5303284416  0.5738236482  0.5776226209  0.6764204747  0.6769292662  0.6211626355  0.6187637969  0.6209481435  0.6225979710  1.4082151753  0.6902092707  0.0001935959  5000          0.0047916675

