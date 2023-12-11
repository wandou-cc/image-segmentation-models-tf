from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = '%s_*.tfrecord'
_CLASS_NAMES = [
    'background', 'aeroplane', 'bicycle', 'bir