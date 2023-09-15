"""Contains utilities for downloading and converting datasets."""
from __future__ import absolute_import, division, print_function

import hashlib
import os
import sys
import tarfile

import glob2
import tensorflow as tf

from six.moves import urllib

_RANDOM_SEED = 0


def parse_glob(path):
  """ returns a file path, or an empty string """
  try:
    return glob2.glob(path)[0]
  except IndexError:
    return ""


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def tfrecord(image, mask, height, width, channels):
  """Build an Example proto for an example.

    Args:
      filename: string, path to an image file, e.g., '/path/to/example.JPG'
      image: string, JPEG encoding of RGB image
      height: integer, image height in pixels
      width: integer, image width in pixels
      mask: string, PNG encoding of ground truth image

    Returns:
      Example proto
    """

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/channels': _int64_feature(channels),
      'image/encoded': _bytes_feature(image),
      'image/mask/encoded': _bytes_feature(mask)
  }))
  return example


class ImageReader(object):
  """Helper class that provides TensorFlow image cod