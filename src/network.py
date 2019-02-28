import numpy as np


class Network(object):
  """ General Params:
    X = (batch_size x input_size)
    y_ = (batch_size x output_size) - 1-hot
  """
  def __init__(self, layers, loss):
    self.layers = layers
    self.loss = loss

  def forward_pass(self, X):
    """ Just do a forward pass calculation on the data"""
    forward = X  # this is the activation of layer l=0
    for l in self.layers:
      forward = l.forward_pass(forward)
    return forward

  def backward_pass(self, Z_L, y_):
    """ Do a backward pass based on loss function and
    final layer scores. Populate the layer errors.
    """
    err = self.loss.loss_error(Z_L, y_)
    for l in reversed(self.layers):
      err = l.backward_pass(err)

  def sgd_step(self, X, y_, learning_rate):
    """ Do a gradient step """
    Z_L = self.forward_pass(X) # This also stores Z values
    self.backward_pass(Z_L, y_)

    for l in self.layers:
      l.update_params(learning_rate)

  def classification_error(self, X, y_):
    """ Classification error (1-0 loss) on batch"""
    a_L = self.forward_pass(X) # final layer activation
    y_ = np.array(y_)
    batch_size = y_.shape[0]
    return np.sum(y_[np.arange(batch_size),
                     np.argmax(a_L, axis=1)]) / batch_size

