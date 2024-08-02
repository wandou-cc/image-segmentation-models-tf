#!/bin/bash
# Run training.
python train.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET} \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL} \
  --save_summaries_se