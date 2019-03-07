""" mnist.py: Simple example neural network trained on MNIST."""
import numpy as np

import src.activations as activations
import src.layers as layers
import src.losses as losses
import src.network as network
import src.datasets as datasets


def mnist_model():
    """ Simple dense model for MNIST."""
    layers_ = [
        layers.ConvolutionalLayer(1, 8),
        layers.MaxPoolLayer(),
        layers.ConvolutionalLayer(8, 8),
        layers.MaxPoolLayer(),
        layers.FlattenLayer((7, 7, 8)),
        layers.DenseLayer(32, 7*7*8, activations.Relu),
        layers.DenseLayer(32, 32, activations.Relu),
        layers.DenseLayer(10, 32, activations.Linear)
    ]
    loss = losses.CrossEntropy()
    return network.Network(layers_, loss)


def train_mnist(mnist_path, batch_size, learning_rate):
    """ Train a simple neural network to classify MNIST."""
    data = datasets.MnistLoader(mnist_path, flatten=False)
    model = mnist_model()

    for i, (bxs, bys) in enumerate(data.train_batches(batch_size)):
        model.gradient_step(bxs, bys, learning_rate)

        if i % 50 == 0:
            acc = model.classification_error(*data.test)
            print(f"Iteration {i} Test Accuracy: {acc:7.3f}")

        if i >= 10 * data.epoch_size / batch_size:   # 10 epochs.
            return


if __name__ == '__main__':
    BATCH_SIZE = 100
    LEARNING_RATE = 0.001
    DATA_PATH = "~/data"
    train_mnist(DATA_PATH, BATCH_SIZE, LEARNING_RATE)
