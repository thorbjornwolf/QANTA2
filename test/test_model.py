# Run with `python -m unittest test.test_model`

import unittest

import model

class TestInitialization(unittest.TestCase):

    def test_shapes(self):
        vocab = {'alpha':0, 'bravo':1, 'charlie':2}
        deplist = {'prep':0, 'pop':1}
        d = 5

        q = model.QANTA(d, vocab, deplist)

        self.assertEquals(q.We.shape, (3, 5))
        self.assertEquals(q.Wr.shape, (2, 5, 5))
        self.assertEquals(q.Wv.shape, (5, 5))
        self.assertEquals(q.b.shape, (5, 1))
