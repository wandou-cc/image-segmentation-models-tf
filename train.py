
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('master', '', 'address of the TensorFlow master to use')
flags.DEFINE_string('train_dir', '/tmp/tfmodel/', 'checkpoints and event logs')
flags.DEFINE_integer('num_clones', 1, 'model clones to deploy')
flags.DEFINE_boolean('clone_on_cpu', False, 'Use CPUs to deploy clones')
flags.DEFINE_integer('worker_replicas', 1, 'worker replicas')
flags.DEFINE_integer('num_ps_tasks', 0, 'param servers. If 0, handle locally')
flags.DEFINE_integer('num_readers', 4, 'parallel dataset readers')
flags.DEFINE_integer('num_preprocessing_threads', 4, 'batch data threads')
flags.DEFINE_integer('log_every_n_steps', 10, 'how often logs are print')
flags.DEFINE_integer('save_summaries_secs', 600, 'summaries saved every x sec')
flags.DEFINE_integer('save_interval_secs', 600, 'model saved every x sec')
flags.DEFINE_integer('task', 0, 'Task id of the replica running the training')

# Optimization Flags
flags.DEFINE_float('weight_decay', 0.00004, 'weight decay on the model weights')
flags.DEFINE_string('optimizer', 'rmsprop', '"adadelta", "adagrad", "adam",'
                    '"ftrl", "momentum", "sgd" or "rmsprop"')
flags.DEFINE_float('adadelta_rho', 0.95, 'decay rate for adadelta')
flags.DEFINE_float('adagrad_initial_accumulator_value', 0.1, 'initial AdaGrad')
flags.DEFINE_float('adam_beta1', 0.9, 'exp. decay for 1st moment estimates')
flags.DEFINE_float('adam_beta2', 0.999, 'exp. decay for 2nd moment estimates')
flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for optimizer')
flags.DEFINE_float('ftrl_learning_rate_power', -0.5, 'learning rate power')
flags.DEFINE_float('ftrl_initial_accumulator_value', 0.1, 'initital FTRL')
flags.DEFINE_float('ftrl_l1', 0.0, 'FTRL l1 regularization strength')
flags.DEFINE_float('ftrl_l2', 0.0, 'FTRL l2 regularization strength')
flags.DEFINE_float('momentum', 0.9, 'MomentumOptimizer and RMSPropOptimizer')
flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum')
flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp')

# Learning rate Flags
flags.DEFINE_string('learning_rate_decay_type', 'polynomial',
                    'exponential/fixed/polynomial')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate')
flags.DEFINE_float('end_learning_rate', 0.0001, 'min end LR polynomial decay')
flags.DEFINE_float('label_smoothing', 0.0, 'amount of label smoothing')
flags.DEFINE_float('learning_rate_decay_factor', 0.94, 'Learning rate decay')
flags.DEFINE_float('num_epochs_per_decay', 2.0, 'epochs when LR decays')
flags.DEFINE_bool('sync_replicas', False, 'synchronize the replicas?')
flags.DEFINE_integer('replicas_to_aggregate', 1, 'gradients before updating')
flags.DEFINE_float('moving_average_decay', None, 'If None,not used')

# Dataset Flags
flags.DEFINE_string('dataset_name', 'imagenet', 'dataset to load')
flags.DEFINE_string('dataset_split_name', 'train', 'name of train/test split')
flags.DEFINE_string('dataset_dir', None, 'where dataset files are stored')
flags.DEFINE_integer('labels_offset', 0, 'Labels offset; used in VGG/ResNet')
flags.DEFINE_string('model_name', 'inception_v3', 'architecture to train')
flags.DEFINE_string('preprocessing_name', None, 'If `None`, model_name is used')
flags.DEFINE_integer('batch_size', 32, 'samples in each batch')
flags.DEFINE_integer('train_image_size', None, 'Train image size')
flags.DEFINE_integer('max_number_of_steps', None, 'maximum training steps')

# Fine-Tuning Flags
flags.DEFINE_string('checkpoint_path', None, 'path to a checkpoint to finetune')
flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint')
flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train'
    'By default, None would train all the variables')
flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables')

FLAGS = flags.FLAGS


def _configure_learning_rate(num_samples_per_epoch, global_step):
  """Configures the learning rate.

      Args:
        num_samples_per_epoch: The samples in each epoch of training.
        global_step: The global_step tensor.

      Returns:
        A `Tensor` representing the learning rate.

      Raises:
        ValueError: if
      """
  decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                    FLAGS.num_epochs_per_decay)
  if FLAGS.sync_replicas:
    decay_steps /= FLAGS.replicas_to_aggregate

  if FLAGS.learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(
        FLAGS.learning_rate,
        global_step,
        decay_steps,
        FLAGS.learning_rate_decay_factor,
        staircase=True,
        name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(
        FLAGS.learning_rate,
        global_step,
        decay_steps,
        FLAGS.end_learning_rate,
        power=1.0,
        cycle=False,
        name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s]',
                     FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
  """Configures the optimizer used for training.

      Args: learning_rate: A scalar or `Tensor` learning rate.

      Returns: An instance of an optimizer.

      Raises:
        ValueError: if FLAGS.optimizer is not recognized.
      """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate, rho=FLAGS.adadelta_rho, epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum=FLAGS.momentum, name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.rmsprop_momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
  return optimizer


def _add_variables_summaries(learning_rate):
  summaries = []
  for variable in slim.get_model_variables():