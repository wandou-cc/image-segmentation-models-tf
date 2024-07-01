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
    # 'resnet_v1_50': resnet_v1.resnet_v1_50,
    # 'resnet_v1_101': resnet_v1.resnet_v1_101,
    # 'resnet_v1_152': resnet_v1.resnet_v1_152,
    # 'resnet_v1_200': resnet_v1.resnet_v1_200,
    # 'resnet_v2_50': resnet_v2.resnet_v2_50,
    # 'resnet_v2_101': resnet_v2.resnet_v2_101,
    # 'resnet_v2_152': resnet_v2.resnet_v2_152,
    # 'resnet_v2_200': resnet_v2.resnet_v2_200,
}

arg_scopes_map = {
    # custom
    'fcn_32':
        fcn.fcn_arg_scope,
    # included in slim contrib
    # 'alexnet_v2': alexnet.alexnet_v2_arg_scope,
    # 'overfeat': overfeat.overfeat_arg_scope,
    # 'vgg_a': vgg.vgg_arg_scope,
    # 'vgg_16': vgg.vgg_arg_scope,
    # 'vgg_19': vgg.vgg_arg_scope,
    # 'inception_v1': inception.inception_v3_arg_scope,
    # 'inception_v2': inception.inception_v3_arg_scope,
    # 'inception_v3': inception.inception_v3_arg_scope,
    # 'resnet_v1_50': resnet_v1.resnet_arg_scope,
    # 'resnet_v