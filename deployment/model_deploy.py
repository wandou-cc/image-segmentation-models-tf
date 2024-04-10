
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
      the tf.GraphKeys.LOSSES collection.

      To recover the losses, summaries or update_ops created by the clone use:
      ```python
        losses = tf.get_collection(tf.GraphKeys.LOSSES, clone.scope)
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, clone.scope)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, clone.scope)
      ```

      The deployment options are specified by the config object and support
      deploying one or several clones on different GPUs and one or several replicas
      of such clones.

      The argument `model_fn` is called `config.num_clones` times to create the
      model clones as `model_fn(*args, **kwargs)`.

      If `config` specifies deployment on multiple replicas then the default
      tensorflow device is set appropriatly for each call to `model_fn` and for the
      slim variable creation functions: model and global variables will be created
      on the `ps` device, the clone operations will be on the `worker` device.

      Args:
        config: A DeploymentConfig object.
        model_fn: A callable. Called as `model_fn(*args, **kwargs)`
        args: Optional list of arguments to pass to `model_fn`.
        kwargs: Optional list of keyword arguments to pass to `model_fn`.

      Returns:
        A list of namedtuples `Clone`.
      """
    clones = []
    args = args or []
    kwargs = kwargs or {}
    with slim.arg_scope(
        [slim.model_variable, slim.variable], device=config.variables_device()):
        # Create clones.
        for i in range(0, config.num_clones):
            with tf.name_scope(config.clone_scope(i)) as clone_scope:
                clone_device = config.clone_device(i)
                with tf.device(clone_device):
                    with tf.variable_scope(
                            tf.get_variable_scope(), reuse=True if i > 0 else
                            None):
                        outputs = model_fn(*args, **kwargs)
                    clones.append(Clone(outputs, clone_scope, clone_device))
    return clones


def _gather_clone_loss(clone, num_clones, regularization_losses):
    """Gather the loss for a single clone.

      Args:
        clone: A Clone namedtuple.
        num_clones: The number of clones being deployed.
        regularization_losses: Possibly empty list of regularization_losses
          to add to the clone losses.

      Returns:
        A tensor for the total loss for the clone.  Can be None.
      """
    # The return value.
    sum_loss = None
    # Individual components of the loss that will need summaries.
    clone_loss = None
    regularization_loss = None
    # Compute and aggregate losses on the clone device.
    with tf.device(clone.device):
        all_losses = []
        clone_losses = tf.get_collection(tf.GraphKeys.LOSSES, clone.scope)
        if clone_losses:
            clone_loss = tf.add_n(clone_losses, name='clone_loss')
            if num_clones > 1:
                clone_loss = tf.div(clone_loss,
                                    1.0 * num_clones,
                                    name='scaled_clone_loss')
            all_losses.append(clone_loss)
        if regularization_losses:
            regularization_loss = tf.add_n(
                regularization_losses, name='regularization_loss')
            all_losses.append(regularization_loss)
        if all_losses:
            sum_loss = tf.add_n(all_losses)
    # Add the summaries out of the clone device block.
    if clone_loss is not None:
        tf.summary.scalar(clone.scope + '/clone_loss', clone_loss)
    if regularization_loss is not None:
        tf.summary.scalar('regularization_loss', regularization_loss)
    return sum_loss


def _optimize_clone(optimizer, clone, num_clones, regularization_losses,
                    **kwargs):
    """Compute losses and gradients for a single clone.

      Args:
        optimizer: A tf.Optimizer  object.
        clone: A Clone namedtuple.
        num_clones: The number of clones being deployed.
        regularization_losses: Possibly empty list of regularization_losses
          to add to the clone losses.
        **kwargs: Dict of kwarg to pass to compute_gradients().

      Returns:
        A tuple (clone_loss, clone_grads_and_vars).
          - clone_loss: A tensor for the total loss for the clone.  Can be None.
          - clone_grads_and_vars: List of (gradient, variable) for the clone.
            Can be empty.
      """
    sum_loss = _gather_clone_loss(clone, num_clones, regularization_losses)
    clone_grad = None
    if sum_loss is not None:
        with tf.device(clone.device):
            clone_grad = optimizer.compute_gradients(sum_loss, **kwargs)
    return sum_loss, clone_grad


def optimize_clones(clones, optimizer, regularization_losses=None, **kwargs):
    """Compute clone losses and gradients for the given list of `Clones`.

      Note: The regularization_losses are added to the first clone losses.

      Args:
       clones: List of `Clones` created by `create_clones()`.
       optimizer: An `Optimizer` object.
       regularization_losses: Optional list of regularization losses. If None it
         will gather them from tf.GraphKeys.REGULARIZATION_LOSSES. Pass `[]` to
         exclude them.
       **kwargs: Optional list of keyword arguments to pass to `compute_gradients`.
