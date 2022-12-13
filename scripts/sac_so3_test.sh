#!/bin/bash

set -e  # exit if any command fails

DATASET_FOLDER=/fabi_project/data/ttl_anat_priors
WORK_DATASET_FOLDER=/fabi_project/data/ttl_anat_priors/fabi_tests

TTL_ROOT=/fabi_project/p3_tract_ttl

mkdir -p $WORK_DATASET_FOLDER

TEST_SUBJECT_ID=fibercup
SUBJECT_ID=fibercup
WORK_EXPERIMENTS_FOLDER=${WORK_DATASET_FOLDER}/experiments
SCORING_DATA=${DATASET_FOLDER}/raw_tournier_basis/${TEST_SUBJECT_ID}/scoring_data

dataset_file=$WORK_DATASET_FOLDER/raw_tournier_basis/${SUBJECT_ID}/${SUBJECT_ID}.hdf5
test_reference_file=$WORK_DATASET_FOLDER/raw_tournier_basis/${TEST_SUBJECT_ID}/masks/${TEST_SUBJECT_ID}_wm.nii.gz

EXPERIMENT=SACSo3FiberCupTrain

# source datafiles
FOD_RAW=$DATASET_FOLDER/raw_tournier_basis/${SUBJECT_ID}/fodfs/fibercup_fodf.nii.gz
WM_RAW=$DATASET_FOLDER/raw_tournier_basis/${SUBJECT_ID}/masks/fibercup_wm.nii.gz
GM_RAW=$DATASET_FOLDER/raw_tournier_basis/${SUBJECT_ID}/masks/fibercup_gm.nii.gz
CSF_RAW=$DATASET_FOLDER/raw_tournier_basis/${SUBJECT_ID}/masks/fibercup_csf.nii.gz
INTERFACE_RAW=$DATASET_FOLDER/raw_tournier_basis/${SUBJECT_ID}/maps/interface.nii.gz


# ID=$(date +"%F-%H_%M_%S")
# ID="2022-11-09-13_06_33"
# ID="2022-11-07-14_33_28"
ID="2022-11-16-15_11_52"

seeds=(1111)

for rng_seed in "${seeds[@]}"
do
  DEST_FOLDER="$WORK_EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$rng_seed"/"test_out"
  HYPERPARAMS="$WORK_EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$rng_seed"/"model"/"hyperparameters.json"
  MODEL="$WORK_EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$rng_seed"/"model"

  mkdir -p $DEST_FOLDER

  FOD_TOURNIER=$DEST_FOLDER/fibercup_tournier07_fodf.nii.gz

  # convert basis
  python /opt/conda/bin/scil_convert_sh_basis.py $FOD_RAW $FOD_TOURNIER 'descoteaux07' -f

  for angle in $(seq 325 1 325)  # $(seq 0 5 360)
  do

  ANGLEDIR="$DEST_FOLDER"/angle_$angle
  mkdir -p $ANGLEDIR
  ROTMAT="$DEST_FOLDER"/angle_$angle/rotmat.txt
  CENTERMAT="$DEST_FOLDER"/center_mat.txt

  # create rotmat
  python /fabi_project/p3_tract_ttl/fabi_scripts/create_rotmat.py $ROTMAT $angle 96 96

  # move to center
  mrtransform -interp cubic -linear $ROTMAT -modulate fod -reorient_fod True -template $FOD_TOURNIER -force $FOD_TOURNIER $ANGLEDIR/fibercup_fodf_tournier.nii.gz
  python /opt/conda/bin/scil_convert_sh_basis.py $ANGLEDIR/fibercup_fodf_tournier.nii.gz $ANGLEDIR/fibercup_fodf.nii.gz 'tournier07' -f
  mrtransform -interp nearest -linear $ROTMAT -template $WM_RAW -force $WM_RAW $ANGLEDIR/fibercup_wm.nii.gz
  mrtransform -interp nearest -linear $ROTMAT -template $GM_RAW -force $GM_RAW $ANGLEDIR/fibercup_gm.nii.gz
  mrtransform -interp nearest -linear $ROTMAT -template $CSF_RAW -force $CSF_RAW $ANGLEDIR/fibercup_csf.nii.gz
  mrtransform -interp nearest -linear $ROTMAT -template $INTERFACE_RAW -force $INTERFACE_RAW $ANGLEDIR/interface.nii.gz
  sh2peaks -num 5 -force $ANGLEDIR/fibercup_fodf_tournier.nii.gz $ANGLEDIR/peaks.nii.gz  # was without tournier before

  python /fabi_project/p3_tract_ttl/TrackToLearn/datasets/create_dataset.py \
    $ANGLEDIR/fibercup_fodf.nii.gz $ANGLEDIR/fibercup_wm.nii.gz \
    $ANGLEDIR/peaks.nii.gz \
    fibercup \
    fibercup \
    $ANGLEDIR \
    --wm $ANGLEDIR/fibercup_wm.nii.gz \
    --gm $ANGLEDIR/fibercup_gm.nii.gz \
    --csf $ANGLEDIR/fibercup_csf.nii.gz \
    --interface $ANGLEDIR/interface.nii.gz \
    --basis_to_tournier

  python $TTL_ROOT/TrackToLearn/runners/so3_test.py \
    "$ANGLEDIR" \
    "$EXPERIMENT" \
    "$ID" \
    "$ANGLEDIR"/fibercup.hdf5 \
    "${SUBJECT_ID}" \
    "$ANGLEDIR"/fibercup_gm.nii.gz \
    "$MODEL" \
    "$HYPERPARAMS" \
    --interface_seeding \
    --n_seeds_per_voxel 32

    # --remove_invalid_streamlines \

  done

done
