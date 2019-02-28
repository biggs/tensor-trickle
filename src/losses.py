import numpy as np

class CrossEntropy(object):
  """ Softmax + CrossEntropyLoss:
  combine for implementation clarity
  """
  def softmax(self,in_vect):
    """ Softmax a vector
    (n,) -> (n,)
    """
    numer = np.exp(in_vect - np.max(in_vect, axis=1, keepdims=True))
    return (numer / np.sum(numer, axis=1, keepdims=True))

  def calc(self, X, y_):
    """ Actual loss from batch
    made unusual choice to sum, not average, so LR scaled by batch size
    X = (batch_size x input_size)
    y_ = (batch_size x output_size) - 1-hot
    -> loss
    """
    softX = self.softmax(X)
    losses = softX * y_ # since 1-hot
    return -np.log(np.sum(losses[:]))

  def loss_error(self, Z_L, y_):
    """ Take the loss on the final scores and get errors to pass backward
    Z_L = (batch_size x output_size)
    y_ = (batch_size x output_size) - 1-hot
    -> loss_error = (batch_size x output_size)
    """
    return self.softmax(Z_L) - y_
