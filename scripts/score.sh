#!/bin/bash

set -e

DATASET_FOLDER=

SUBJECT_ID=$3
SCORING_DATA=${DATASET_FOLDER}/raw/${SUBJECT_ID}/scoring_data

mkdir -p $2

python scripts/score_tractogram.py $1 \
  "$SCORING_DATA" \
  $2 \
  --save_full_vc \
  --save_full_ic \
  --save_full_nc \
  --save_ib \
  --save_vb -f -v
