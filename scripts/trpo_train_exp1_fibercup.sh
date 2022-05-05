#!/bin/bash

DATASET_FOLDER=
WORK_DATASET_FOLDER=
mkdir -p $WORK_DATASET_FOLDER

TEST_SUBJECT_ID=fibercup
SUBJECT_ID=fibercup
EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
WORK_EXPERIMENTS_FOLDER=${WORK_DATASET_FOLDER}/experiments
SCORING_DATA=${WORK_DATASET_FOLDER}/raw/${TEST_SUBJECT_ID}/scoring_data

mkdir -p $WORK_DATASET_FOLDER/raw

echo "Transfering data to working folder..."
cp -rn ${DATASET_FOLDER}/raw/${SUBJECT_ID} ${WORK_DATASET_FOLDER}/raw/
cp -rn ${DATASET_FOLDER}/raw/${TEST_SUBJECT_ID} ${WORK_DATASET_FOLDER}/raw/

dataset_file=$WORK_DATASET_FOLDER/raw/${SUBJECT_ID}/${SUBJECT_ID}.hdf5
test_dataset_file=$WORK_DATASET_FOLDER/raw/${TEST_SUBJECT_ID}/${TEST_SUBJECT_ID}.hdf5
test_reference_file=$WORK_DATASET_FOLDER/raw/${TEST_SUBJECT_ID}/masks/${TEST_SUBJECT_ID}_wm.nii.gz

max_ep=1000 # Chosen empirically
log_interval=50 # Log at n episodes
lr=0.0005 # Learning rate
gamma=0.9 # Gamma for reward discounting
action_std=0.0

delta=0.01
K_epochs=1
max_backtracks=10
lmbda=0.95
n_update=100
entropy_loss_coeff=0.0
backtrack_coeff=0.5

valid_noise=0.0 # Noise to add to make a prob output. 0 for deterministic

n_seeds_per_voxel=2 # Seed per voxel
max_angle=20 # Maximum angle for streamline curvature
tracking_batch_size=4096

EXPERIMENT=TRPO_FiberCupTrain075

ID=$(date +"%F-%H_%M_%S")

seeds=(1111 2222 3333 4444 5555)
rng_seed=${seeds[$SLURM_ARRAY_TASK_ID]}

DEST_FOLDER="$WORK_EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$rng_seed"

python TrackToLearn/runners/trpo_train.py \
  $DEST_FOLDER \
  "$EXPERIMENT" \
  "$ID" \
  "${dataset_file}" \
  "${SUBJECT_ID}" \
  "${test_dataset_file}" \
  "${TEST_SUBJECT_ID}" \
  "${test_reference_file}" \
  "${SCORING_DATA}" \
  --max_ep=${max_ep} \
  --log_interval=${log_interval} \
  --lr=${lr} \
  --gamma=${gamma} \
  --action_std=${action_std} \
  --delta=${delta} \
  --K_epochs=${K_epochs} \
  --max_backtracks=${max_backtracks} \
  --lmbda=${lmbda} \
  --n_update=${n_update} \
  --entropy_loss_coeff=${entropy_loss_coeff} \
  --backtrack_coeff=${backtrack_coeff} \
  --rng_seed=${rng_seed} \
  --n_seeds_per_voxel=${n_seeds_per_voxel} \
  --max_angle=${max_angle} \
  --tracking_batch_size=${tracking_batch_size} \
  --use_gpu \
  --use_comet \
  --run_tractometer

mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"
mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"/"$ID"
cp -f -r $DEST_FOLDER "$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/

