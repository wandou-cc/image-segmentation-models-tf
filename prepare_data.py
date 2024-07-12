
"""Downloads and converts datatsets."""
from __future__ import absolute_import, division, print_function

import tensorflow as tf

from datasets import (download_and_convert_pascal)

FLAGS = tf.app.flags.FLAGS
flags = tf.app.flags
flags.DEFINE_string('dataset_name', None, '"pascal"')
flags.DEFINE_string('dataset_dir', None, 'where to put TFRecords ')
flags.DEFINE_string('dataset_archive', None, 'where to find archives')


def main(_):
  if not FLAGS.dataset_name: