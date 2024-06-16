import numpy as np


def get_kernel_size(factor):
  """Find the kernel size given the desired factor of upsampling."""
  return 2 * factor - factor % 2


def upsample_filt(size):
  """Make a 2D bilinear kernel suitable for upsa