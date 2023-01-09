#!/bin/bash


set -e  # exit if any command fails

DATASET_FOLDER=/fabi_project/data/ttl_anat_priors
WORK_DATASET_FOLDER=/fabi_project/data/ttl_anat_priors/fabi_tests

TTL_ROOT=/fabi_project/p3_tract_ttl

mkdir -p $WORK_DATASET_FOLDER

TEST_SUBJECT_ID=fibercup
SUBJECT_ID=fibercup
EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
WORK_EXPERIMENTS_FOLDER=${WORK_DATASET_FOLDER}/experiments
SCORING_DATA=${DATASET_FOLDER}/raw/${TEST_SUBJECT_ID}/scoring_data

echo "Transfering data to working folder..."
mkdir -p $WORK_DATASET_FOLDER/raw/${SUBJECT_ID}
cp -rn $DATASET_FOLDER/raw/${SUBJECT_ID} $WORK_DATASET_FOLDER/raw/

dataset_file=$WORK_DATASET_FOLDER/raw/${SUBJECT_ID}/${SUBJECT_ID}.hdf5
test_dataset_file=$WORK_DATASET_FOLDER/raw/${TEST_SUBJECT_ID}/${TEST_SUBJECT_ID}.hdf5
test_reference_file=$WORK_DATASET_FOLDER/raw/${TEST_SUBJECT_ID}/masks/${TEST_SUBJECT_ID}_wm.nii.gz

max_ep=500 # Chosen empirically
log_interval=50 # Log at n steps
lr=0.005 # Learning rate
gamma=0.75 # Gamma for reward discounting
alpha=0.5

valid_noise=0.0 # Noise to add to make a prob output. 0 for deterministic

n_seeds_per_voxel=2 # Seed per voxel
max_angle=30 # Maximum angle for streamline curvature

EXPERIMENT=SACAutoFiberCupTrain

# ID=$(date +"%F-%H_%M_%S")
ID="_debug_log"

# seeds=(1111 2222 3333 4444 5555)
seeds=(1111)

for rng_seed in "${seeds[@]}"
do

  DEST_FOLDER="$WORK_EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$rng_seed"

  python $TTL_ROOT/TrackToLearn/runners/sac_auto_train.py \
    "$DEST_FOLDER" \
    "$EXPERIMENT" \
    "$ID" \
    "${dataset_file}" \
    "${SUBJECT_ID}" \
    "${test_dataset_file}" \
    "${TEST_SUBJECT_ID}" \
    "${test_reference_file}" \
    --ground_truth_folder="${SCORING_DATA}" \
    --max_ep=${max_ep} \
    --log_interval=${log_interval} \
    --lr=${lr} \
    --gamma=${gamma} \
    --alpha=${alpha} \
    --rng_seed=${rng_seed} \
    --n_seeds_per_voxel=${n_seeds_per_voxel} \
    --max_angle=${max_angle} \
    --valid_noise=$valid_noise \
    --use_gpu \
    --use_comet \
    --run_tractometer \
    --interface_seeding
    # --render

  mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"
  mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"/"$ID"
  mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"/"$ID"/
  cp -f -r $DEST_FOLDER "$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/

done
