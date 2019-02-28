""" activations.py: Activation functions for neural network layers."""
import numpy as np


class Activation(object):
    """ Abstract class for differentiable activation."""

    def calc(self, preactiv):
        """ Returns array after activation application."""
        raise NotImplementedError

    def deriv(self, preactiv):
        """ Returns array activation derivative."""
        raise NotImplementedError


class Linear(Activation):
    """ Differentiable linear 'activation'."""

    def calc(self, preactiv):
        return preactiv

    def deriv(self, preactiv):
        return 1.


class Relu(Activation):
    """ Differentiable ReLu activation."""

    def calc(self, preactiv):
        return np.maximum(0, preactiv)

    def deriv(self, preactiv):
        return (preactiv > 0) * 1.
