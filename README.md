# Tensor-Trickle

#### A simple, easily understandable library for deep learning in Python with NumPy.

Ever wondered what the internals of your PyTorch or TensorFlow model look like? Tensor-Trickle works in a similar way, providing a high-level Keras-based API but explicitly implementing backpropogation (no autodiff) in dense, pooling, and convolutional layers.

It is written for pedalogical clarity, containing no fancy features like GPU acceleration, but achieves a reasonable enough performance for experimentation.

See `mnist.py` for a simple example usage. Nothing here should look too different from any other deep learning framework. Source code for the library lives in `src`.


#### Requirements

Python3, NumPy, observations.

[Observations](http://edwardlib.org/api/observations) provides a simple API for loading common datasets. It can be easily installed with `pip3 install observations`.
