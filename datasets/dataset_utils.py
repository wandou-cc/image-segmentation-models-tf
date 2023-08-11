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