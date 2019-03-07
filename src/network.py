""" network.py: Contains Network general model class."""
import numpy as np


class Network(object):
    """ DNN grouping together multiple layers.

    Attributes:
        layers: a list of layers.Layer derived objects
        loss: a loss function object
    """

    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

    def forward_pass(self, input_):
        """ Forward pass storing layer outputs.

        Args:
            input_: network input [batch_size, input_size].

        Returns:
            output of the neural network [batch_size, output_size]
            where output_size is the number of units of the final layer.
        """
        forward = input_
        for layer in self.layers:
            forward = layer.forward_pass(forward)
        return forward

    def backward_pass(self, out, label):
        """ Backward pass populating layer errors.

        Args:
            label: (1-hot) network output [batch_size, output_size].
        """
        err = self.loss.loss_error(out, label)
        for layer in reversed(self.layers):
            err = layer.backward_pass(err)

    def gradient_step(self, input_, label, learning_rate):
        """ Parameter gradient step based on errors."""
        out = self.forward_pass(input_)
        self.backward_pass(out, label)
        for layer in self.layers:
            layer.update_params(learning_rate)

    def classification_error(self, input_, label):
        """ Return whole-batch classification error (1-0 loss)."""
        batch_size = label.shape[0]
        out = self.forward_pass(input_)
        return np.mean(
            label[np.arange(batch_size), np.argmax(out, axis=1)])
