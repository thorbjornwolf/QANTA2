import numpy as np

def dtanh(x):
    """derivative of normalized tanh
    Taken directly from original QANTA implementation
    I have not figured out why this works
    """
    norm = np.linalg.norm(x)
    y = x - np.power(x, 3)
    dia = np.diag((1 - np.square(x)).flatten()) / norm
    pro = y.dot(x.T) / np.power(norm, 3)
    out = dia - pro
    return out

def normalize(vec):
    return vec / np.linalg.norm(vec)

class Adagrad(object):
    """Container for Adagrad parameters for a single
    entity, such as We or Wv"""

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.reset()

    def reset(self):
        self.sum_of_squared_gradients = None

    def get_scale(self, gradients):
        """Returns the scaling factor for adagrad for the given
        gradients and their history (self.sum_of_squared_gradients)
        """
        if self.sum_of_squared_gradients is None:
            self.sum_of_squared_gradients = np.square(gradients)
            # Initialize with a small value to avoid initial division by zero
            small = self.sum_of_squared_gradients < 1e-3
            self.sum_of_squared_gradients[small] = 1e-3
        self.sum_of_squared_gradients += np.square(gradients)
        denominators = np.sqrt(self.sum_of_squared_gradients)
        return self.learning_rate / denominators
