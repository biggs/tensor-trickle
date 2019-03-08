""" dense.py: Dense neural network layer."""
import logging
import numpy as np

from src.layers.base import Layer



class DenseLayer(Layer):
    """ Fully connected neural network layer.

    Attributes:
        weights: the matrix of weights for the layer [no_units, input_size]
        bias: a bias vector [no_units,]
        activation: an activation function object
        input_: stored forward pass input [batch_size, input_size]
        preactiv: stored pre-activation output [batch_size, no_units]
        preact_err: Calculated pre-activation error [batch_size, no_units]
    """

    def __init__(self, no_units, input_size, activation):
        """ Xavier initialisation for weights, bias to more standard zeros."""
        xav = np.sqrt(6. / (no_units+input_size))
        self.weights = np.random.uniform(
            low=-xav, high=xav, size=(no_units, input_size))
        self.bias = np.zeros(no_units)
        self.activation = activation()

    def forward_pass(self, input_):
        """ Calculate and store forward pass, returning output.

        Args:
            input_: Layer input [batch_size, input_size]

        Returns:
            Layer output [batch_size, no_units]
        """
        self.input_ = input_
        self.preactiv = input_ @ self.weights.T + self.bias
        return self.activation.calc(self.preactiv)

    def backward_pass(self, err):
        """ Store layer error and return error of previous layer

        Args:
            err: Layer error passed back [batch_size, no_units]

        Returns:
            Layer input error [batch_size, input_size]
        """
        self.preact_err = err * self.activation.deriv(self.preactiv)
        return self.preact_err @ self.weights

    def update_params(self, learning_rate):
        """ After passes use stored error to update weights and biases. """
        self.bias -= learning_rate * np.sum(self.preact_err, axis=0)
        self.weights -= learning_rate * self.preact_err.T @ self.input_  # No divide.
