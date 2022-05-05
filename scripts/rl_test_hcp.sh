#!/bin/bash

set -e

DATASET_FOLDER=

EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
SCORING_DATA=${DATASET_FOLDER}/raw/${SUBJECT_ID}/scoring_data

dataset_file=$DATASET_FOLDER/raw/${SUBJECT_ID}/${SUBJECT_ID}.hdf5
reference_file=$DATASET_FOLDER/raw/${SUBJECT_ID}/masks/${SUBJECT_ID}_wm.nii.gz

step_size=0.375 # Step size (in mm) # Half ISMRM2015 step size to cover the same

tracking_batch_size=250000
n_seeds_per_voxel=2
min_length=20
max_length=200

EXPERIMENT=$1
ID=$2

validstds=(0.2)
subjectids=(hcp_100206)
seeds=(1111 2222 3333 4444 5555)

for SEED in "${seeds[@]}"
do
  for SUBJECT_ID in "${subjectids[@]}"
  do
    for valid_noise in "${validstds[@]}"
    do
      EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
      SCORING_DATA=${DATASET_FOLDER}/raw/${SUBJECT_ID}/scoring_data
      DEST_FOLDER="$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$SEED"

      dataset_file=$DATASET_FOLDER/raw/${SUBJECT_ID}/${SUBJECT_ID}.hdf5
      reference_file=$DATASET_FOLDER/raw/${SUBJECT_ID}/masks/${SUBJECT_ID}_wm.nii.gz

      echo $DEST_FOLDER/model/hyperparameters.json
      python TrackToLearn/runners/test.py \
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
        --test_max_angle="30" \
        --use_gpu \
        --fa_map="$DATASET_FOLDER"/raw/${SUBJECT_ID}/dti/"${SUBJECT_ID}"_fa.nii.gz \
        --remove_invalid_streamlines

      test_folder=$DEST_FOLDER/scoring_"${valid_noise}"_"${SUBJECT_ID}"_${n_seeds_per_voxel}

      mkdir -p $test_folder

      mv $DEST_FOLDER/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk $test_folder/

    done
  done
done
