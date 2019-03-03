import numpy as np
from src.layers.base import Layer
from src.layers.utils import max_pool
from src.layers.utils import max_pool_indices
from src.layers.utils import distribute_to_indices


class MaxPoolLayer(Layer):
    """ 2x2 Max Pooling Layer with stride of 2.

    Max pools and stores indices for backward pass. On forward:
    input - [batch_size, width, height, depth]
    output - [batch_size, width / 2, height / 2, depth]

    Attributes:
        shape: most recent forward pass shape.
        indices: stored indices for backward pass.
    """

    def forward_pass(self, input_):
        """ Pass forward, storing which indices were passed."""
        self.shape = input_.shape
        self.indices = max_pool_indices(input_)
        return max_pool(input_)

    def backward_pass(self, err):
        """ Pass back only to indices passed forward."""
        return distribute_to_indices(self.indices, err, self.shape)

    def update_params(self, _):
        pass
