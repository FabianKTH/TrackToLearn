#!/bin/bash

DATASET_FOLDER=
WORK_DATASET_FOLDER=
mkdir -p $WORK_DATASET_FOLDER

TEST_SUBJECT_ID=ismrm2015
SUBJECT_ID=ismrm2015
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
lr=5.0e-5 # Learning rate
gamma=0.85 # Gamma for reward discounting
alpha=0.1

valid_noise=0.0 # Noise to add to make a prob output. 0 for deterministic

n_seeds_per_voxel=2 # Seed per voxel
max_angle=30 # Maximum angle for streamline curvature
tracking_batch_size=4096

EXPERIMENT=SAC_ISMRM2015Train075

ID=$(date +"%F-%H_%M_%S")

seeds=(1111 2222 3333 4444 5555)
rng_seed=${seeds[$SLURM_ARRAY_TASK_ID]}

DEST_FOLDER="$WORK_EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$rng_seed"

python TrackToLearn/runners/sac_train.py \
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
  --alpha=${alpha} \
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

