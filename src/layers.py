""" layers.py: differentiable, trainable neural network layers."""
import logging
import numpy as np

from src.utils import np_str


class Layer(object):
    """ Abstract base class for differentiable, trainable layer."""

    def forward_pass(self, input_):
        """ Forward pass returning and storing outputs."""
        raise NotImplementedError()

    def backward_pass(self, back):
        """ Backward pass returning and storing errors."""
        raise NotImplementedError()

    def update_params(self, learning_rate):
        """ Update layer weights based on stored errors."""
        raise NotImplementedError()



class DenseLayer(Layer):
    """ Fully connected neural network layer.

    Attributes:
        weights: the matrix of weights for the layer [no_units, input_size]
        bias: a bias vector [no_units,]
        activation: an activation function object
        input_: stored forward pass input [batch_size, input_size]
        preactiv: stored forward pass output pre-activation [batch_size, no_units]
        E: Calculated pre-activation error TODO? [batch_size, no_units]
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
        self.E = err * self.activation.deriv(self.preactiv)
        return self.E @ self.weights

    def update_params(self, learning_rate):
        """ After passes use stored error to update weights and biases. """
        d_bias = np.sum(self.E, axis=0)
        d_weights = self.E.T @ self.input_    # no divide as def
        logging.debug("weights update =\n%s", np_str(d_weights))
        logging.debug("bias update =\n%s", np_str(d_bias))
        self.bias -= learning_rate * d_bias
        self.weights -= learning_rate * d_weights
