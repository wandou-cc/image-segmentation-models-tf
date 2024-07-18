
"""Provides utilities to preprocess images for the Inception networks."""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops


def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

      Args:
        x: input Tensor.
        func: Python function to apply.
        num_cases: Python int32, number of cases to sample sel from.

      Returns: