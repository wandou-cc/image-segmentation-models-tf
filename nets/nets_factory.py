"""Contains a factory for building various models."""

from __future__ import absolute_import, division, print_function

import functools

import tensorflow as tf
import tensorflow.contrib.slim as slim
# from tensorflow.contrib.slim.nets import (alexnet, inception, overfeat, resnet_v1, resnet_v2, vgg)
from nets import fcn

slim = tf.contrib.slim

networks_map = {
    'fcn_32':
        fcn.fcn_32,
    # included in slim contrib
    # 'alexnet_v2': alexnet.alexnet_v2,
    # 'overfeat': overfeat.overfeat,
    # 'vgg_a': vgg.vgg_a,
    # 'vgg_16': vgg.vgg_16,
    # 'vgg_19': vgg.vgg_19,
    # 'inception_v1': inception.inception_v1,
    # 'inception_v2': inception.inception_v2,
    # 'inception_v3': inception.inception_v3,
    # 'resnet_v1_50': r