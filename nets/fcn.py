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
      weights_regularizer=slim.l2_regularizer(weight_decay),
      biases_initializer=tf.zeros_initializer):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc


def fcn_32(inputs,
           num_classes=21,
           is_training=True,
           dropout_prob=0.5,
           scope='fcn_32'):
  with tf.variable_scope(scope, 'fcn_32', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected, conv2d_transpose and max_pool2d.
    with slim.arg_scope(
        [
            slim.conv2d, slim.fully_connected, slim.max_pool2d,
            slim.conv2d_transpose
        ],
        outputs_collections=end_points_collection):

      # Contracting portion is VGG-16 https://goo.gl/dM7PWe 
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      pool1 = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(pool1, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      pool2 = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(pool2, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      pool3 = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(pool3, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      pool4 = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(pool4