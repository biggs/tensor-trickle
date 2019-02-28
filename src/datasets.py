""" datasets.py: Load in data for training."""
import numpy as np
from observations import mnist


def generator(arrays, batch_size):
    """Generate batches, one with respect to each array's first axis."""
    starts = [0] * len(arrays)  # pointers to where we are in iteration
    while True:
        batches = []
        for i, array in enumerate(arrays):
            start = starts[i]
            stop = start + batch_size
            diff = stop - array.shape[0]
            if diff <= 0:
                batch = array[start:stop]
                starts[i] += batch_size
            else:
                batch = np.concatenate((array[start:], array[:diff]))
                starts[i] = diff
            batches.append(batch)
        yield batches


def one_hot(array):
    """ From vector of batch numbers to one-hot version."""
    ret = np.zeros((array.size, array.max()+1))
    ret[np.arange(array.size), array] = 1.
    return ret

def binarize(array):
    """ Binarize a NumPy array in range 0-255."""
    return (array > (255/2)).astype(np.int)


class MnistLoader(object):
    """ Load and pre-process MNIST."""

    def __init__(self, data_path):
        """ Take in a path to (down)load data."""
        (train_x, train_y), (test_x, test_y) = mnist(data_path)
        self.train = (binarize(train_x), one_hot(train_y))
        self.test = (binarize(test_x), one_hot(test_y))
        self.batch_size = 0

    def train_batches(self, batch_size):
        """ Return batch generator of batch_size."""
        if not self.batch_size == batch_size:
            self.batches = generator(list(self.train), batch_size)
        return self.batches
