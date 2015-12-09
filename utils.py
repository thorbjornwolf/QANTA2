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


def find_missing(data, process, lo=0, hi=None, n_missing=None):
    """Use case: process(data) returns too few elements. Which are missing?
    This method will tell you. 

    data is a listlike with order preserved
    process is a method that can take data as an argument, and returns a 
        listlike. It is assumed to be deterministic in its error.
    lo is the lowest index (inclusive) for where the error(s) can be.
        Default is 0.
    hi is the highest index (exclusive) for where the error(s) can be.
        Default is len(data)

    Returns a list of indices of the data points that disappear.
    If nothing disappears, the list is empty.
    """
    if hi is None:
        hi = len(data)

    if hi <= lo:
        return []

    if n_missing is None:
        d = data[lo:hi]
        res = list(process(d))
    
        n_missing = len(d) - len(res)

        # None missing?
        if n_missing == 0:
            return []

        # All missing?
        if n_missing == len(d):
            return range(lo, hi)

    # Binary search
    midway = ((hi-lo)/2) + lo

    # Get indices of missing items in lower half of data slice
    lo_miss = find_missing(data, process, lo, midway)

    # Can we exit without checking higher half?
    if len(lo_miss) == n_missing:
        return lo_miss

    # Get indices of missing items in upper half of data slice
    hi_miss = find_missing(data, process, midway, hi)

    return lo_miss + hi_miss

def split(data, n_slices=None):
    """returns a generator of n_slices data slices, with
    data slice length varying by at most 1 (if the data 
    cannot be evenly split), and always so that the
    first slices are 1 longer or of equal length to the last
    slices.

    Data order is retained, and itertools.chain(split(data, n)) == data

    Example:
        >>> list(split(['a', 'b', 'c', 'd'], 3))
        [['a', 'b'], ['c'], ['d']]
        >>> list(split(['a', 'b', 'c', 'd', 'e', 'f'], 2))
        [['a', 'b', 'c'], ['d', 'e', 'f']]
        >>> list(split(['a', 'b', 'c', 'd', 'e', 'f'], 4))
        [['a', 'b'], ['c', 'd'], ['e'], ['f']]
    """
    
    a,b = divmod(len(data), n_slices)
    if a == 0: # data size smaller than the wanted number of slices
        return ((d,) for d in data)

    # chunk sizes
    cs = b*[a+1] + ((len(data) - b*(a+1))/a)*[a]

    # chunk indices in data
    ci = [sum(cs[:i]) for i in xrange(len(cs))] + [sum(cs)]

    return (data[ci[i]:ci[i+1]] for i in xrange(len(ci) - 1))

class Vocabulary(dict):
    """dict extension for word-to-index mapping.
    If queried for a word it doesn't contain, returns
    its own length. That way, unknown words are mapped
    to the highest index.
    """
    def __getitem__(self, key):
        return self.get(key, super(Vocabulary, self).__len__())

    def __len__(self):
        return super(Vocabulary, self).__len__() + 1