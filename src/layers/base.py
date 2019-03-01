""" base.py: Base class for differentiable neural network layers."""

class Layer(object):
    """ Abstract base class for differentiable layer."""

    def forward_pass(self, input_):
        """ Forward pass returning and storing outputs."""
        raise NotImplementedError()

    def backward_pass(self, err):
        """ Backward pass returning and storing errors."""
        raise NotImplementedError()

    def update_params(self, learning_rate):
        """ Update layer weights based on stored errors."""
        raise NotImplementedError()
