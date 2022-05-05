# This script tracks with all the trained models according to their experiments
set -e

# exp1

# ./scripts/rl_test_fibercup.sh
# ./scripts/rl_test_fibercup.sh
# ./scripts/rl_test_fibercup.sh
# ./scripts/rl_test_fibercup.sh
# ./scripts/rl_test_fibercup.sh
# ./scripts/rl_test_fibercup.sh DDPG_FiberCupTrain075 2021-09-15-09_27_05
# ./scripts/rl_test_fibercup.sh TD3_FiberCupTrain075 2021-09-15-09_37_15
# ./scripts/rl_test_fibercup.sh SAC_FiberCupTrain075 2021-09-08-10_43_20
# ./scripts/rl_test_fibercup.sh SAC_Auto_FiberCupTrain075 2021-09-16-10_40_46
# 
# # ./rl_test_ismrm2015.sh
# # ./rl_test_ismrm2015.sh
# # ./rl_test_ismrm2015.sh
# # ./rl_test_ismrm2015.sh
# ./scripts/rl_test_ismrm2015.sh DDPG_ISMRM2015Train075 2021-09-28-09_18_04
# ./scripts/rl_test_ismrm2015.sh TD3_ISMRM2015Train075 2021-09-26-14_12_06
# ./scripts/rl_test_ismrm2015.sh SAC_ISMRM2015Train075 2021-09-23-23_32_06
./scripts/rl_test_ismrm2015.sh SAC_Auto_ISMRM2015Train075 2021-09-16-10_40_57
# 
# # exp2
# ./rl_test_fibercup.sh VPG_FiberCupTrainGM075
./scripts/rl_test_fibercup.sh A2C_FiberCupTrainGM075 2021-10-27-11_42_14
./scripts/rl_test_fibercup.sh ACKTR_FiberCupTrainGM075 2021-10-01-08_52_48
./scripts/rl_test_fibercup.sh TRPO_FiberCupTrainGM075 2021-10-01-08_47_24
./scripts/rl_test_fibercup.sh PPO_FiberCupTrainGM075 2021-10-13-11_13_26
./scripts/rl_test_fibercup.sh DDPG_FiberCupTrainGM075 2021-08-26-08_12_07
./scripts/rl_test_fibercup.sh TD3_FiberCupTrainGM075 2021-08-26-08_12_02
./scripts/rl_test_fibercup.sh SAC_FiberCupTrainGM075 2021-08-26-08_12_05
./scripts/rl_test_fibercup.sh SAC_Auto_FiberCupTrainGM075 2021-08-26-09_42_22

# ./rl_test_ismrm2015.sh
# ./rl_test_ismrm2015.sh
# ./rl_test_ismrm2015.sh
# ./rl_test_ismrm2015.sh
./scripts/rl_test_ismrm2015.sh DDPG_ISMRM2015TrainGM075 2021-09-07-15_17_09
./scripts/rl_test_ismrm2015.sh TD3_ISMRM2015TrainGM075 2021-09-07-17_26_59
./scripts/rl_test_ismrm2015.sh SAC_ISMRM2015TrainGM075 2021-09-10-08_58_04
./scripts/rl_test_ismrm2015.sh SAC_Auto_ISMRM2015TrainGM075 2021-08-26-09_50_01
