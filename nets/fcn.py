""" Fully Convolutional Models for Semantic Segmentation
    arXiv:1605.06211
    https://github.com/shelhamer/fcn.berkeleyvision.org
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf

slim = tf.contrib.slim


def fcn_arg_scope(weight_decay=0.0005):
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected, slim.conv2d_transpose],
      activation_fn=tf.nn.relu,
      weights_regularizer=slim.l