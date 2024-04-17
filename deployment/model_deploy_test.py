
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