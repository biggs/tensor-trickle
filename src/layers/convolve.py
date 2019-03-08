""" convolve.py: Convolutional neural network layers."""
import numpy as np
from src.layers.base import Layer
from src.layers.utils import convolve_3x3
from src.layers.utils import get_3x3_patches


class ConvolutionalLayer(Layer):
    """ 3x3 Convolutional layer with stride of 1.

    input - [batch_size, width, height, in_depth]
    output - [batch_size, width, height, out_depth]

    Attributes:
        weights: filter weights [3, 3, in_depth, out_depth].
        input_: stored inputs [batch_size, width, height, in_depth]
        err: stored error [batch_size, width, height, out_depth]
    """

    def __init__(self, in_depth, out_depth):
        """ Uses Xavier initialisation. No biases."""
        xav = np.sqrt(6. / (9 * (in_depth + out_depth)))
        self.weights = np.random.uniform(size=(3, 3, in_depth, out_depth),
                                         low=-xav, high=xav)

    def forward_pass(self, input_):
        """ Store input and return convolution with filters.

        Args:
            input_: layer inputs [batch_size, width, height, in_depth]
        Returns:
            Layer output [batch_size, width, height, out_depth]
        """
        self.input_ = input_
        return convolve_3x3(input_, self.weights)

    def backward_pass(self, err):
        """ Store layer error and return error of previous layer

        Args:
            err: Passed back error [batch_size, width, height, out_depth]

        Returns:
            Layer input error [batch_size, width, height, input_depth]
        """
        self.err = err
        reverse_weights = self.weights[::-1, ::-1].transpose(0, 1, 3, 2)
        return convolve_3x3(err, reverse_weights)

    def update_params(self, learning_rate):
        """ Update filter weights using stored error and inputs."""
        # n: batch_size, w: width, h: height, a,b,c,d: weight dimensions.
        input_patches = get_3x3_patches(self.input_)
        grad = np.einsum("nwhabc,nwhd->abcd", input_patches, self.err)
        self.weights -= learning_rate * grad
