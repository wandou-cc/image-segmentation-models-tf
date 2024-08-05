
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