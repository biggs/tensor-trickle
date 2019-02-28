import logging
import numpy as np

from src.utils import np_str


class Layer(object):
  def forward_pass(self, A_prev):
    raise NotImplementedError()

  def backward_pass(self, E):
    raise NotImplementedError()

  def update_params(self,learning_rate):
    pass



class DenseLayer(Layer):
  def __init__(self, no_units, input_size, activation):
    """ Xavier initialisation for W, b to more standard zeros"""
    xav = np.sqrt(6./(no_units+input_size))
    self.W = np.random.uniform(
      low=-xav, high=xav, size=(no_units,input_size))
    self.b = np.zeros(no_units)
    self.activation = activation

  def forward_pass(self, forward):
    """ Calculate and *store* forward pass
    (using activation prev layer):
    forward (batch_size x input_size) -> A (batch_size x no_units)
    *Store*: forward, Z, Z_grad
    """
    Z = forward @ self.W.T + self.b
    logging.debug("Z = %s", np_str(Z))

    self.forward = forward  # store input from prev layer
    self.Z = Z  # store Z for this
    self.Z_grad = self.activation.deriv(Z) # this layer activation grad
    return self.activation.calc(Z)  # pass forward activation

  def backward_pass(self, back):
    """ *store* layer error and pass back error to previous layer
    back (batch_size x no_units) -> new_back (batch_size x input_size)
    """
    E = back * self.Z_grad
    self.E = E   # store this layer err
    return E @ self.W  # to pass back further

  def update_params(self, learning_rate):
    """ Uses stored self.err to update weights and biases
    *Must run forward and backward passes first*
    """
    db = np.sum(self.E, axis=0)
    dW = self.E.T @ self.forward
    logging.debug("dW update =\n%s", np_str(dW))
    logging.debug("db update =\n%s", np_str(db))
    self.b -= learning_rate * db # no divide as def
    self.W -= learning_rate * dW

