
"""Deploy Slim models across multiple clones and replicas.

# TODO(sguada) docstring paragraph by (a) motivating the need for the file and
# (b) defining clones.

# TODO(sguada) describe the high-level components of model deployment.
# E.g. "each model deployment is composed of several parts: a DeploymentConfig,
# which captures A, B and C, an input_fn which loads data.. etc

To easily train a model on multiple GPUs or across multiple machines this
module provides a set of helper functions: `create_clones`,
`optimize_clones` and `deploy`.

Usage:

  g = tf.Graph()

  # Set up DeploymentConfig
  config = model_deploy.DeploymentConfig(num_clones=2, clone_on_cpu=True)

  # Create the global step on the device storing the variables.
  with tf.device(config.variables_device()):
    global_step = slim.create_global_step()

  # Define the inputs
  with tf.device(config.inputs_device()):
    images, labels = LoadData(...)
    inputs_queue = slim.data.prefetch_queue((images, labels))

  # Define the optimizer.
  with tf.device(config.optimizer_device()):
    optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, FLAGS.momentum)

  # Define the model including the loss.
  def model_fn(inputs_queue):
    images, labels = inputs_queue.dequeue()
    predictions = CreateNetwork(images)
    slim.losses.log_loss(predictions, labels)

  model_dp = model_deploy.deploy(config, model_fn, [inputs_queue],
                                 optimizer=optimizer)

  # Run training.
  slim.learning.train(model_dp.train_op, my_log_dir,
                      summary_op=model_dp.summary_op)

The Clone namedtuple holds together the values associated with each call to
model_fn:
  * outputs: The return values of the calls to `model_fn()`.
  * scope: The scope used to create the clone.
  * device: The device used to create the clone.

DeployedModel namedtuple, holds together the values needed to train multiple
clones:
  * train_op: An operation that run the optimizer training op and include
    all the update ops created by `model_fn`. Present only if an optimizer
    was specified.
  * summary_op: An operation that run the summaries created by `model_fn`
    and process_gradients.
  * total_loss: A `Tensor` that contains the sum of all losses created by
    `model_fn` plus the regularization losses.
  * clones: List of `Clone` tuples returned by `create_clones()`.

DeploymentConfig parameters:
  * num_clones: Number of model clones to deploy in each replica.
  * clone_on_cpu: True if clones should be placed on CPU.
  * replica_id: Integer.  Index of the replica for which the model is
      deployed.  Usually 0 for the chief replica.
  * num_replicas: Number of replicas to use.
  * num_ps_tasks: Number of tasks for the `ps` job. 0 to not use replicas.
  * worker_job_name: A name for the worker job.
  * ps_job_name: A name for the parameter server job.

TODO(sguada):
  - describe side effect to the graph.
  - what happens to summaries and update_ops.
  - which graph collections are altered.
  - write a tutorial on how to use this.
  - analyze the possibility of calling deploy more than once.


"""

from __future__ import absolute_import, division, print_function

import collections

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

slim = tf.contrib.slim

__all__ = [
    'create_clones',
    'deploy',
    'optimize_clones',
    'DeployedModel',
    'DeploymentConfig',
    'Clone',
]

# Namedtuple used to represent a clone during deployment.
Clone = collections.namedtuple(
    'Clone',
    [
        'outputs',  # Whatever model_fn() returned.
        'scope',  # The scope used to create it.
        'device',  # The device used to create.
    ])

# Namedtuple used to represent a DeployedModel, returned by deploy().
DeployedModel = collections.namedtuple(
    'DeployedModel',
    [
        'train_op',  # The `train_op`
        'summary_op',  # The `summary_op`
        'total_loss',  # The loss `Tensor`
        'clones',  # A list of `Clones` tuples.
    ])

# Default parameters for DeploymentConfig
_deployment_params = {
    'num_clones': 1,
    'clone_on_cpu': False,
    'replica_id': 0,
    'num_replicas': 1,
    'num_ps_tasks': 0,
    'worker_job_name': 'worker',
    'ps_job_name': 'ps'
}


def create_clones(config, model_fn, args=None, kwargs=None):
    """Creates multiple clones according to config using a `model_fn`.

      The returned values of `model_fn(*args, **kwargs)` are collected along with
      the scope and device used to created it in a namedtuple
      `Clone(outputs, scope, device)`

      Note: it is assumed that any loss created by `model_fn` is collected at