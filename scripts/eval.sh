#!/bin/bash

#Run evaluation.
python eval.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET} \
  --dataset_dir=${DA