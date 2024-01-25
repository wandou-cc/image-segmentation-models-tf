from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = '%s_*.tfrecord'
_CLASS_NAMES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'potted-plant', 'sheep', 'sofa', 'train', 'tv/monitor',
    'ambigious'
]

SPLITS_TO_SIZES = {'train': 3320, 'validation': 350}

_NUM_CLASSES = 21

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'Ground truth segmentation mask',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split na