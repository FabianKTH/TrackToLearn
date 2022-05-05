#!/bin/bash


set -e  # exit if any command fails

DATASET_FOLDER=
HOME_DATASET_FOLDER=

TEST_SUBJECT_ID=fibercup
SUBJECT_ID=fibercup
EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
HOME_EXPERIMENTS_FOLDER=${HOME_DATASET_FOLDER}/experiments
SCORING_DATA=${DATASET_FOLDER}/raw/${TEST_SUBJECT_ID}/scoring_data

mkdir -p $HOME_DATASET_FOLDER/raw/${SUBJECT_ID}

echo "Transfering data to working folder..."
cp -r ${DATASET_FOLDER}/raw/${SUBJECT_ID} ${HOME_DATASET_FOLDER}/raw/
cp -r ${DATASET_FOLDER}/raw/${TEST_SUBJECT_ID} ${HOME_DATASET_FOLDER}/raw/

dataset_file=$HOME_DATASET_FOLDER/raw/${SUBJECT_ID}/${SUBJECT_ID}.hdf5
test_dataset_file=$HOME_DATASET_FOLDER/raw/${TEST_SUBJECT_ID}/${TEST_SUBJECT_ID}.hdf5
test_reference_file=$HOME_DATASET_FOLDER/raw/${TEST_SUBJECT_ID}/masks/${TEST_SUBJECT_ID}_wm.nii.gz

max_ep=1000 # Chosen empirically
log_interval=50 # Log at n steps
lr=0.00005 # Learning rate
gamma=0.85 # Gamma for reward discounting


valid_noise=0.0 # Noise to add to make a prob output. 0 for deterministic

n_seeds_per_voxel=2 # Seed per voxel
max_angle=30 # Maximum angle for streamline curvature
tracking_batch_size=50000
length_weighting=0.1

EXPERIMENT=SAC_Auto_FiberCupTrainLength075

ID=$(date +"%F-%H_%M_%S")

seeds=(1111 2222 3333 4444 5555)

for rng_seed in "${seeds[@]}"
do

  DEST_FOLDER="$HOME_EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$rng_seed"

  python TrackToLearn/runners/sac_auto_train.py \
    "$DEST_FOLDER" \
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
    --rng_seed=${rng_seed} \
    --n_seeds_per_voxel=${n_seeds_per_voxel} \
    --max_angle=${max_angle} \
    --valid_noise=$valid_noise \
    --length_weighting=${length_weighting} \
    --use_gpu \
    --use_comet \
    --run_tractometer \
    --tracking_batch_size=${tracking_batch_size}
    # --render

    mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"
    mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"/"$ID"
    cp -f -r $DEST_FOLDER "$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/

done
