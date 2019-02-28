import sys
import logging

import numpy as np
from observations import mnist

import src.activations as activations
import src.layers as layers
import src.losses as losses
import src.network as network


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
    """ [batch_size] -> [batch_size x no_classes]"""
    ret = np.zeros((array.size, array.max()+1))
    ret[np.arange(array.size), array] = 1.
    return ret

def binarize(array):
    return (array > (255/2)).astype(np.int)


def load_mnist():
    train, test = mnist(DATA_PATH)
    train = (binarize(train[0]), one_hot(train[1]))
    test = (binarize(test[0]), one_hot(test[1]))
    return train, test


def main():
    train, test = load_mnist()
    layers_ = [
        layers.DenseLayer(32, 28*28, activations.relu),
        layers.DenseLayer(32, 32, activations.relu),
        layers.DenseLayer(10, 32, activations.linear)
    ]
    loss = losses.CrossEntropy()
    model = network.Network(layers_, loss)

    for i, (xs, ys) in enumerate(generator(list(train), BATCH_SIZE)):
        model.sgd_step(xs, ys, LEARNING_RATE)

        if i % 500 == 0:
            acc = model.classification_error(*test)
            print(f"Iteration {i} Test Accuracy: {acc:7.3f}")


if __name__ == '__main__':
    DEBUG = False
    logging.basicConfig(
      stream=sys.stderr, level=(logging.WARNING if not DEBUG else logging.DEBUG))
    NUM_EPOCHS = 1
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    DATA_PATH = "~/data"
    main()
