#!/bin/bash

set -e  # exit if any command fails

DATASET_FOLDER=/fabi_project/data/ttl_anat_priors
DEST_FOLDER=/fabi_project/data/ttl_anat_priors/fabi_tests/dsets

mkdir -p $DEST_FOLDER

SUBJECT_ID=fibercup

dataset_file=$WORK_DATASET_FOLDER/raw/${SUBJECT_ID}/${SUBJECT_ID}.hdf5

# source datafiles
FOD_RAW=$DATASET_FOLDER/raw/${SUBJECT_ID}/fodfs/fibercup_fodf.nii.gz
PEAKS_RAW=$DATASET_FOLDER/raw/${SUBJECT_ID}/fodfs/fibercup_peaks.nii.gz
WM_RAW=$DATASET_FOLDER/raw/${SUBJECT_ID}/masks/fibercup_wm.nii.gz
GM_RAW=$DATASET_FOLDER/raw/${SUBJECT_ID}/masks/fibercup_gm.nii.gz
CSF_RAW=$DATASET_FOLDER/raw/${SUBJECT_ID}/masks/fibercup_csf.nii.gz
INTERFACE_RAW=$DATASET_FOLDER/raw/${SUBJECT_ID}/maps/interface.nii.gz


python /fabi_project/p3_tract_ttl/TrackToLearn/datasets/create_dataset.py \
  $FOD_RAW $WM_RAW \
  $PEAKS_RAW \
  fibercup \
  fibercup \
  $DEST_FOLDER \
  --wm $WM_RAW \
  --gm $GM_RAW \
  --csf $CSF_RAW \
  --interface $INTERFACE_RAW \
  --basis_to_tournier