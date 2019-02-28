""" losses.py: neural network loss functions."""
import numpy as np


def softmax(array):
    """ Stable softmax for numpy array."""
    numer = np.exp(array - np.max(array, axis=1, keepdims=True))
    return numer / np.sum(numer, axis=1, keepdims=True)


class CrossEntropy(object):
    """ Cross Entropy Loss."""

    def calc(self, input_, label):
        """ Returns batch loss.

        Args:
            input_: [batch_size x input_size]
            label: [batch_size x output_size] (1-hot)

        Returns:
            scalar cross-entropy loss, summed over batch
            (scaling learning rate).
        """
        losses = softmax(input_) * label  # 1-hot
        return -np.log(np.sum(losses[:]))

    def loss_error(self, out, label):
        """ Error to pass backward for training.

        Args:
            out: (batch_size x output_size)
            label: (batch_size x output_size) - 1-hot

        Returns:
            the error loss to pass back [batch_size x output_size]
        """
        return softmax(out) - label
