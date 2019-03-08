""" flatten.py: Flattening neural network layer."""
from src.layers.base import Layer


class FlattenLayer(Layer):
    """ Flattening neural network layer.

    Flattens non-batch dimensions. On forward:
    input - [batch_size, width, height, depth]
    output - [batch_size, width * height * depth]

    Attributes:
        shape: input shape [width, height, depth]
    """

    def __init__(self, shape):
        self.shape = shape

    def forward_pass(self, input_):
        """ Flatten non-batch dimensions."""
        return input_.reshape((input_.shape[0], -1))

    def backward_pass(self, err):
        """ Reshape passed back error to self.shape."""
        return err.reshape((-1,) + self.shape)

    def update_params(self, _):
        pass
