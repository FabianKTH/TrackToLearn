#!/bin/bash


if [ -z "$1" ]; then
    echo "Missing experiment name"
fi

DATASET_FOLDER=
HOME_DATASET_FOLDER=

TEST_SUBJECT_ID=fibercup
SUBJECT_ID=fibercup
EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
HOME_EXPERIMENTS_FOLDER=${HOME_DATASET_FOLDER}/experiments
SCORING_DATA=${DATASET_FOLDER}/raw/${TEST_SUBJECT_ID}/scoring_data

mkdir -p $HOME_DATASET_FOLDER/raw/${SUBJECT_ID}

echo "Transfering data to working folder..."
cp -rn ${DATASET_FOLDER}/raw/${SUBJECT_ID} ${HOME_DATASET_FOLDER}/raw/
cp -rn ${DATASET_FOLDER}/raw/${TEST_SUBJECT_ID} ${HOME_DATASET_FOLDER}/raw/

dataset_file=$HOME_DATASET_FOLDER/raw/${SUBJECT_ID}/${SUBJECT_ID}_mask.hdf5
test_dataset_file=$HOME_DATASET_FOLDER/raw/${TEST_SUBJECT_ID}/${TEST_SUBJECT_ID}_mask.hdf5
test_reference_file=$HOME_DATASET_FOLDER/raw/${TEST_SUBJECT_ID}/masks/${TEST_SUBJECT_ID}_wm.nii.gz

max_ep=100 # Chosen empirically
log_interval=100 # Log at n steps
action_std=0.45 # STD deviation for action
lr=3.39e-05 # Learning rate
gamma=0.93 # Gamma for reward discounting
lmbda=0.9 # Lambda for advantage
rng_seed=4444 # Seed for general randomness

training_batch_size=16384

n_latent_var=2048 # Layer width
add_neighborhood=1.2 # Neighborhood to add to state input
valid_noise=0.1 # Noise to add to make a prob output. 0 for deterministic
tracking_batch_size=50000

n_seeds_per_voxel=2 # Seed per voxel
max_angle=60 # Maximum angle for streamline curvature
min_length=20 # Minimum streamline length
max_length=200 # Maximum streamline length
step_size=0.75 # Step size (in mm)
n_signal=1 # Use last n input
n_dirs=4 # Also input last n directions taken

alignment_weighting=1. # Reward weighting for alignment
straightness_weighting=0. # Reward weighting for sinuosity
length_weighting=0.0 # Reward weighting for length
target_bonus_factor=0.0 # Reward penalty/bonus for end-of-trajectory actions
exclude_penalty_factor=0.0 # Reward penalty/bonus for end-of-trajectory actions
angle_penalty_factor=0.0 # Reward penalty/bonus for end-of-trajectory actions

EXPERIMENT=TD3FibercupRewardTest 
ID=$(date +"%F-%H_%M_%S")
NAME=$EXPERIMENT/$ID

mkdir -p $HOME_EXPERIMENTS_FOLDER/$EXPERIMENT/$ID/masks
mkdir -p $HOME_EXPERIMENTS_FOLDER/$EXPERIMENT/$ID/model
mkdir -p $HOME_EXPERIMENTS_FOLDER/$EXPERIMENT/$ID/plots

cp -f -r $HOME_DATASET_FOLDER/raw/$SUBJECT_ID/masks $HOME_EXPERIMENTS_FOLDER/$EXPERIMENT/$ID
cp -f -r $HOME_DATASET_FOLDER/raw/$SUBJECT_ID/bundles $HOME_EXPERIMENTS_FOLDER/$EXPERIMENT/$ID
cp -f -r $HOME_DATASET_FOLDER/raw/$TEST_SUBJECT_ID/masks $HOME_EXPERIMENTS_FOLDER/$EXPERIMENT/$ID
cp -f -r $HOME_DATASET_FOLDER/raw/$TEST_SUBJECT_ID/bundles $HOME_EXPERIMENTS_FOLDER/$EXPERIMENT/$ID

stds=(0.2)
for std in "${stds[@]}"
do

  python TrackToLearn/runners/td3_train.py $HOME_EXPERIMENTS_FOLDER \
    $EXPERIMENT \
    $ID \
    ${dataset_file} \
    ${SUBJECT_ID} \
    ${test_dataset_file} \
    ${TEST_SUBJECT_ID} \
    ${test_reference_file} \
    ${SCORING_DATA} \
    --max_ep=${max_ep} \
    --log_interval=${log_interval} \
    --action_std=${std} \
    --lr=${lr} \
    --gamma=${gamma} \
    --lmbda=${lmbda} \
    --rng_seed=${rng_seed} \
    --n_seeds_per_voxel=${n_seeds_per_voxel} \
    --max_angle=${max_angle} \
    --min_length=${min_length} \
    --max_length=${max_length} \
    --step_size=${step_size} \
    --n_latent_var=${n_latent_var} \
    --add_neighborhood=${add_neighborhood} \
    --alignment_weighting=${alignment_weighting} \
    --straightness_weighting=${straightness_weighting} \
    --length_weighting=${length_weighting} \
    --target_bonus_factor=${target_bonus_factor} \
    --exclude_penalty_factor=${exclude_penalty_factor} \
    --angle_penalty_factor=${angle_penalty_factor} \
    --tracking_batch_size=${tracking_batch_size} \
    --n_signal=${n_signal} \
    --n_dirs=${n_dirs} \
    --valid_noise=$valid_noise \
    --training_batch_size=${training_batch_size} \
    --load_policy=${load_pretrained_policy} \
    --use_gpu \
    --use_comet \
    --run_tractometer

  mkdir -p $EXPERIMENTS_FOLDER/$EXPERIMENT
  mkdir -p $EXPERIMENTS_FOLDER/$EXPERIMENT/$ID
  cp -f -r $HOME_EXPERIMENTS_FOLDER/$EXPERIMENT/$ID $EXPERIMENTS_FOLDER/$EXPERIMENT/
done
