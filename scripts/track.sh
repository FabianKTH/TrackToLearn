#!/bin/bash

set -e


step_size=0.325 # Step size (in mm)
tracking_batch_size=10000
n_seeds_per_voxel=10
min_length=10
max_length=200

EXPERIMENT=$1
ID=$2

SUBJECT_ID=hcp_100206
valid_noise=0.1

DATASET_FOLDER=

reference_file=$DATASET_FOLDER/raw/${SUBJECT_ID}/masks/${SUBJECT_ID}_wm.nii.gz
signal_file=$DATASET_FOLDER/raw/${SUBJECT_ID}/hcp_100206_signal.nii.gz
peaks_file=$DATASET_FOLDER/raw/${SUBJECT_ID}/fodfs/hcp_100206_peaks.nii.gz
seeding_file=$DATASET_FOLDER/raw/${SUBJECT_ID}/maps/hcp_100206_interface.nii.gz
tracking_file=$DATASET_FOLDER/raw/${SUBJECT_ID}/masks/hcp_100206_wm.nii.gz
target_file=$DATASET_FOLDER/raw/${SUBJECT_ID}/masks/hcp_100206_gm.nii.gz
exclude_file=$DATASET_FOLDER/raw/${SUBJECT_ID}/masks/hcp_100206_csf.nii.gz

seeds=(1111 2222 3333 4444 5555)
for SEED in "${seeds[@]}"
do

  EXPERIMENTS_FOLDER=${DATASET_FOLDER}/experiments
  DEST_FOLDER="$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$SEED"
  out_tractogram="$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$SEED"/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk

  python TrackToLearn/runners/track.py \
    "$DEST_FOLDER" \
    "$EXPERIMENT" \
    "$ID" \
    "${signal_file}" \
    "${peaks_file}" \
    "${seeding_file}" \
    "${tracking_file}" \
    "${target_file}" \
    "${exclude_file}" \
    "${SUBJECT_ID}" \
    "${reference_file}" \
    $DEST_FOLDER/model \
    $DEST_FOLDER/model/hyperparameters.json \
    --out_tractogram="${out_tractogram}" \
    --valid_noise="${valid_noise}" \
    --step_size="${step_size}" \
    --n_seeds_per_voxel="${n_seeds_per_voxel}" \
    --tracking_batch_size="${tracking_batch_size}" \
    --min_length="$min_length" \
    --max_length="$max_length" \
    --use_gpu \
    --fa_map="$DATASET_FOLDER"/raw/${SUBJECT_ID}/dti/"${SUBJECT_ID}"_fa.nii.gz \
    --gm_seeding \
    --remove_invalid_streamlines

  test_folder=$DEST_FOLDER/tracking_"${valid_noise}"_"${SUBJECT_ID}"

  mkdir -p $test_folder

  mv $out_tractogram $test_folder/
done

