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

dataset_file=$HOME_DATASET_FOLDER/raw/${SUBJECT_ID}/${SUBJECT_ID}.hdf5
test_dataset_file=$HOME_DATASET_FOLDER/raw/${TEST_SUBJECT_ID}/${TEST_SUBJECT_ID}.hdf5
test_reference_file=$HOME_DATASET_FOLDER/raw/${TEST_SUBJECT_ID}/masks/${TEST_SUBJECT_ID}_wm.nii.gz

max_ep=1000 # Chosen empirically
log_interval=50 # Log at n steps
lr=0.0001 # Learning rate
gamma=0.75 # Gamma for reward discounting

alpha=0.10  # alpha for temperature

valid_noise=0.0 # Noise to add to make a prob output. 0 for deterministic

n_seeds_per_voxel=2 # Seed per voxel
max_angle=30 # Maximum angle for streamline curvature

EXPERIMENT=SAC_FiberCupTrainGM075

ID=$(date +"%F-%H_%M_%S")

seeds=(1111 2222 3333 4444 5555)

for rng_seed in "${seeds[@]}"
do

  DEST_FOLDER="$HOME_EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/"$rng_seed"

  python TrackToLearn/runners/sac_train.py \
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
    --alpha=${alpha} \
    --rng_seed=${rng_seed} \
    --n_seeds_per_voxel=${n_seeds_per_voxel} \
    --max_angle=${max_angle} \
    --valid_noise=$valid_noise \
    --use_gpu \
    --use_comet \
    --run_tractometer \
    --gm_seeding
    # --render

  mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"
  mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"/"$ID"
  mkdir -p $EXPERIMENTS_FOLDER/"$EXPERIMENT"/"$ID"/
  # cp -f -r $DEST_FOLDER "$EXPERIMENTS_FOLDER"/"$EXPERIMENT"/"$ID"/

done
n_seeds_per_voxel=33

validstds=(0.0 0.1 0.2 0.3)
subjectids=(fibercup fibercup_flipped)

for SEED in "${seeds[@]}"
do
  for SUBJECT_ID in "${subjectids[@]}"
  do
    for valid_noise in "${validstds[@]}"
    do

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
        --n_seeds_per_voxel="${n_seeds_per_voxel}" \
        --use_gpu \
        --fa_map="$DATASET_FOLDER"/raw/${SUBJECT_ID}/dti/"${SUBJECT_ID}"_fa.nii.gz \
        --remove_invalid_streamlines

      test_folder=$DEST_FOLDER/scoring_"${valid_noise}"_"${SUBJECT_ID}"

      mkdir -p $test_folder

      mv $DEST_FOLDER/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk $test_folder/

      python scripts/score_tractogram.py $test_folder/tractogram_"${EXPERIMENT}"_"${ID}"_"${SUBJECT_ID}".trk \
        "$SCORING_DATA" \
        $test_folder \
        --save_full_vc \
        --save_full_ic \
        --save_full_nc \
        --save_ib \
        --save_vb -f -v
    done
  done
done

