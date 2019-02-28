import numpy as np


class Activation(object):
  def calc(self, Z):
    raise NotImplementedError

  def deriv(self, Z):
    raise NotImplementedError


class LinearActiv(Activation):
  def calc(self, Z):
    return Z

  def deriv(self, Z):
    return 1.


class ReluActiv(Activation):
  def calc(self, Z):
    return np.maximum(0, Z)

  def deriv(self, Z):
    return (Z > 0) * 1.

relu = ReluActiv()
linear = LinearActiv()
