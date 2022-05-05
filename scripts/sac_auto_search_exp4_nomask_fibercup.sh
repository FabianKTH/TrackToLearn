#!/bin/bash

DATASET_FOLDER=
WORK_DATASET_FOLDER=


TEST_SUBJECT_ID=fibercup_nomask
SUBJECT_ID=fibercup_nomask
EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
WORK_EXPERIMENTS_FOLDER=${WORK_DATASET_FOLDER}/experiments
SCORING_DATA=${DATASET_FOLDER}/raw/${TEST_SUBJECT_ID}/scoring_data

mkdir -p $WORK_DATASET_FOLDER/raw/${SUBJECT_ID}

echo "Transfering data to working folder..."
cp -r ${DATASET_FOLDER}/raw/${SUBJECT_ID} ${WORK_DATASET_FOLDER}/raw/
cp -r ${DATASET_FOLDER}/raw/${TEST_SUBJECT_ID} ${WORK_DATASET_FOLDER}/raw/

dataset_file=$WORK_DATASET_FOLDER/raw/${SUBJECT_ID}/${SUBJECT_ID}.hdf5
test_dataset_file=$WORK_DATASET_FOLDER/raw/${TEST_SUBJECT_ID}/${TEST_SUBJECT_ID}.hdf5
test_reference_file=$WORK_DATASET_FOLDER/raw/${TEST_SUBJECT_ID}/masks/${TEST_SUBJECT_ID}_wm.nii.gz

max_ep=1000 # Chosen empirically
log_interval=50 # Log at n steps

valid_noise=0.0 # Noise to add to make a prob output. 0 for deterministic

n_seeds_per_voxel=2 # Seed per voxel
max_angle=30 # Maximum angle for streamline curvature
n_dirs=4
tracking_batch_size=50000

EXPERIMENT=SAC_Auto_FiberCupSearchNoMask075

ID=$(date +"%F-%H_%M_%S")

rng_seed=1111

DEST_FOLDER="$WORK_EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$rng_seed"

export COMET_OPTIMIZER_ID=c1d5b23b75b04c67a82001b0da48a312

python TrackToLearn/searchers/sac_auto_searcher.py \
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
  --rng_seed=${rng_seed} \
  --n_seeds_per_voxel=${n_seeds_per_voxel} \
  --max_angle=${max_angle} \
  --valid_noise=$valid_noise \
  --n_dirs=${n_dirs} \
  --tracking_batch_size=${tracking_batch_size} \
  --use_gpu \
  --use_comet \
  --run_tractometer
  # --render

mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"
mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"/"$ID"
mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"/"$ID"/"$rng_seed"
cp -f -r $DEST_FOLDER "$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/
