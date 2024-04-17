
"""Tests for model_deploy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from deployment import model_deploy

slim = tf.contrib.slim


class DeploymentConfigTest(tf.test.TestCase):

    def testDefaults(self):
        deploy_config = model_deploy.DeploymentConfig()

        self.assertEqual(slim.get_variables(), [])
        self.assertEqual(deploy_config.caching_device(), None)
        self.assertDeviceEqual(deploy_config.clone_device(0), '')
        self.assertEqual(deploy_config.clone_scope(0), '')
        self.assertDeviceEqual(deploy_config.optimizer_device(), 'CPU:0')
        self.assertDeviceEqual(deploy_config.inputs_device(), 'CPU:0')
        self.assertDeviceEqual(deploy_config.variables_device(), 'CPU:0')

    def testCPUonly(self):
        deploy_config = model_deploy.DeploymentConfig(clone_on_cpu=True)

        self.assertEqual(deploy_config.caching_device(), None)
        self.assertDeviceEqual(deploy_config.clone_device(0), 'CPU:0')
        self.assertEqual(deploy_config.clone_scope(0), '')
        self.assertDeviceEqual(deploy_config.optimizer_device(), 'CPU:0')
        self.assertDeviceEqual(deploy_config.inputs_device(), 'CPU:0')
        self.assertDeviceEqual(deploy_config.variables_device(), 'CPU:0')

    def testMultiGPU(self):
        deploy_config = model_deploy.DeploymentConfig(num_clones=2)

        self.assertEqual(deploy_config.caching_device(), None)
        self.assertDeviceEqual(deploy_config.clone_device(0), 'GPU:0')
        self.assertDeviceEqual(deploy_config.clone_device(1), 'GPU:1')
        self.assertEqual(deploy_config.clone_scope(0), 'clone_0')
        self.assertEqual(deploy_config.clone_scope(1), 'clone_1')
        self.assertDeviceEqual(deploy_config.optimizer_device(), 'CPU:0')
        self.assertDeviceEqual(deploy_config.inputs_device(), 'CPU:0')
        self.assertDeviceEqual(deploy_config.variables_device(), 'CPU:0')

    def testPS(self):
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=1, num_ps_tasks=1)

        self.assertDeviceEqual(deploy_config.clone_device(0), '/job:worker')
        self.assertEqual(deploy_config.clone_scope(0), '')
        self.assertDeviceEqual(deploy_config.optimizer_device(),
                               '/job:worker/device:CPU:0')
        self.assertDeviceEqual(deploy_config.inputs_device(),
                               '/job:worker/device:CPU:0')
        with tf.device(deploy_config.variables_device()):
            a = tf.Variable(0)
            b = tf.Variable(0)
            c = tf.no_op()
            d = slim.variable(
                'a', [], caching_device=deploy_config.caching_device())
        self.assertDeviceEqual(a.device, '/job:ps/task:0/device:CPU:0')
        self.assertDeviceEqual(a.device, a.value().device)
        self.assertDeviceEqual(b.device, '/job:ps/task:0/device:CPU:0')
        self.assertDeviceEqual(b.device, b.value().device)
        self.assertDeviceEqual(c.device, '')
        self.assertDeviceEqual(d.device, '/job:ps/task:0/device:CPU:0')
        self.assertDeviceEqual(d.value().device, '')

    def testMultiGPUPS(self):
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=2, num_ps_tasks=1)

        self.assertEqual(deploy_config.caching_device()(tf.no_op()), '')
        self.assertDeviceEqual(
            deploy_config.clone_device(0), '/job:worker/device:GPU:0')
        self.assertDeviceEqual(
            deploy_config.clone_device(1), '/job:worker/device:GPU:1')
        self.assertEqual(deploy_config.clone_scope(0), 'clone_0')
        self.assertEqual(deploy_config.clone_scope(1), 'clone_1')
        self.assertDeviceEqual(deploy_config.optimizer_device(),
                               '/job:worker/device:CPU:0')
        self.assertDeviceEqual(deploy_config.inputs_device(),
                               '/job:worker/device:CPU:0')

    def testReplicasPS(self):
        deploy_config = model_deploy.DeploymentConfig(
            num_replicas=2, num_ps_tasks=2)

        self.assertDeviceEqual(deploy_config.clone_device(0), '/job:worker')
        self.assertEqual(deploy_config.clone_scope(0), '')
        self.assertDeviceEqual(deploy_config.optimizer_device(),
                               '/job:worker/device:CPU:0')
        self.assertDeviceEqual(deploy_config.inputs_device(),
                               '/job:worker/device:CPU:0')

    def testReplicasMultiGPUPS(self):
        deploy_config = model_deploy.DeploymentConfig(
            num_replicas=2, num_clones=2, num_ps_tasks=2)
        self.assertDeviceEqual(
            deploy_config.clone_device(0), '/job:worker/device:GPU:0')
        self.assertDeviceEqual(
            deploy_config.clone_device(1), '/job:worker/device:GPU:1')
        self.assertEqual(deploy_config.clone_scope(0), 'clone_0')
        self.assertEqual(deploy_config.clone_scope(1), 'clone_1')
        self.assertDeviceEqual(deploy_config.optimizer_device(),
                               '/job:worker/device:CPU:0')
        self.assertDeviceEqual(deploy_config.inputs_device(),
                               '/job:worker/device:CPU:0')

    def testVariablesPS(self):
        deploy_config = model_deploy.DeploymentConfig(num_ps_tasks=2)

        with tf.device(deploy_config.variables_device()):
            a = tf.Variable(0)
            b = tf.Variable(0)
            c = tf.no_op()
            d = slim.variable(
                'a', [], caching_device=deploy_config.caching_device())

        self.assertDeviceEqual(a.device, '/job:ps/task:0/device:CPU:0')
        self.assertDeviceEqual(a.device, a.value().device)
        self.assertDeviceEqual(b.device, '/job:ps/task:1/device:CPU:0')
        self.assertDeviceEqual(b.device, b.value().device)
        self.assertDeviceEqual(c.device, '')
        self.assertDeviceEqual(d.device, '/job:ps/task:0/device:CPU:0')
        self.assertDeviceEqual(d.value().device, '')


def LogisticClassifier(inputs, labels, scope=None, reuse=None):
    with tf.variable_scope(
            scope, 'LogisticClassifier', [inputs, labels], reuse=reuse):
        predictions = slim.fully_connected(
            inputs, 1, activation_fn=tf.sigmoid, scope='fully_connected')
        slim.losses.log_loss(predictions, labels)
        return predictions


def BatchNormClassifier(inputs, labels, scope=None, reuse=None):
    with tf.variable_scope(
            scope, 'BatchNormClassifier', [inputs, labels], reuse=reuse):
        inputs = slim.batch_norm(inputs, decay=0.1)
        predictions = slim.fully_connected(
            inputs, 1, activation_fn=tf.sigmoid, scope='fully_connected')
        slim.losses.log_loss(predictions, labels)
        return predictions
