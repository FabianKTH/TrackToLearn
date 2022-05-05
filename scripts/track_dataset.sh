#!/bin/bash

set -e

DATASET_FOLDER=

dataset_file=$DATASET_FOLDER/raw/${SUBJECT_ID}/${SUBJECT_ID}.hdf5
reference_file=$DATASET_FOLDER/raw/${SUBJECT_ID}/masks/${SUBJECT_ID}_wm.nii.gz

step_size=0.325 # Step size (in mm)
tracking_batch_size=50000
n_seeds_per_voxel=1
min_length=10
max_length=200

EXPERIMENT=$1
ID=$2

SEED=1111
SUBJECT_ID=hcp_100206
valid_noise=0.2

EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
DEST_FOLDER="$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$SEED"

dataset_file=$DATASET_FOLDER/raw/${SUBJECT_ID}/${SUBJECT_ID}.hdf5
reference_file=$DATASET_FOLDER/raw/${SUBJECT_ID}/masks/${SUBJECT_ID}_wm.nii.gz

python -m cProfile -o program.prof TrackToLearn/runners/test.py \
  "$DEST_FOLDER" \
  "$EXPERIMENT" \
  "$ID" \
  "${dataset_file}" \
  "${SUBJECT_ID}" \
  "${reference_file}" \
  "${SCORING_DATA}" \
  $DEST_FOLDER/model \
  $DEST_FOLDER/model/hyperparameters.json \
  --valid_noise="${valid_noise}" \
  --step_size="${step_size}" \
  --n_seeds_per_voxel="${n_seeds_per_voxel}" \
  --tracking_batch_size="${tracking_batch_size}" \
  --min_length="$min_length" \
  --max_length="$max_length" \
  --use_gpu \
  --fa_map="$DATASET_FOLDER"/raw/${SUBJECT_ID}/dti/"${SUBJECT_ID}"_fa.nii.gz \
  --remove_invalid_streamlines

test_folder=$DEST_FOLDER/tracking_"${valid_noise}"_"${SUBJECT_ID}"

mkdir -p $test_folder

mv $DEST_FOLDER/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk $test_folder/
