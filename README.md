# Tensor-Trickle

#### A simple, easily understandable library for deep learning in Python with NumPy.

Ever wondered what the internals of your PyTorch or TensorFlow model look like? It's hard to look inside the code because so many use cases need to be handled.

Tensor-Trickle works in a similar way, providing a high-level Keras-based API but explicitly implementing backpropogation (no autodiff) in dense, pooling, and convolutional layers.

It is written for pedalogical clarity, containing no fancy features like GPU acceleration, but achieves a reasonable enough performance for experimentation.

See `mnist.py` for a simple example usage with digit classification. Nothing here should look too different from any other deep learning framework. Source code for the library lives in `src`.


#### Layout

`mnist.py`: simple example usage.

Inside `src`:
```
network.py: Contains Network general model class.
losses.py: neural network loss functions.
datasets.py: Load in data for training.
activations.py: Activation functions for neural network layers.
```
Inside `src/layers`:
```
layers/base.py: Base class for differentiable neural network layers.
layers/dense.py: Dense neural network layer.
layers/flatten.py: Flattening neural network layer.
layers/pooling.py: Pooling neural network layers.
layers/convolve.py: Convolutional neural network layers.
layers/utils.py: Utilities for convolution and pooling.
```
#### Requirements

Python3, NumPy, observations.

[Observations](http://edwardlib.org/api/observations) provides a simple API for loading common datasets, useful for this and other projects. It can be easily installed with `pip3 install observations`.
