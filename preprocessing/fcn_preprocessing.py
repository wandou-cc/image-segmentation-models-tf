from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from PIL import Image

slim = tf.contrib.slim

# vgg network
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def _mean_image_subtraction(image, means):
  if image.get_shape().ndims