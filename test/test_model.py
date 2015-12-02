# Run with `python -m unittest test.test_model`

import unittest

import numpy as np

import model

class TestNode(object):
    def __init__(self, word_index, children=None, dependency_index=None):
        self.word_index = word_index
        self.children = children or []
        self.dependency_index = dependency_index

class TestInitialization(unittest.TestCase):

    def test_shapes(self):
        vocab = {'alpha':0, 'bravo':1, 'charlie':2}
        deplist = {'prep':0, 'pop':1}
        d = 5
        answers = None

        q = model.QANTA(d, vocab, deplist, answers)

        self.assertEquals(q.We.shape, (3, 5))
        self.assertEquals(q.Wr.shape, (2, 5, 5))
        self.assertEquals(q.Wv.shape, (5, 5))
        self.assertEquals(q.b.shape, (5, 1))

    def test_embedding_calculation(self):
        vocab = {'alpha':0, 'bravo':1}
        deplist = {'prep':0}
        d = 3
        answers = None

        q = model.QANTA(d, vocab, deplist, answers)
        q.We = np.array([[1,2,3], [-1,0,1]])
        q.Wr = np.array([[
            [1,2,3],
            [2,3,1], 
            [3,2,1]
        ]])
        q.Wv = np.array([
            [-1, 2, 2],
            [ 3, 2, 3],
            [ 0, 1, 2]
        ])
        q.b = np.array([-2, -3, 1])


        b = TestNode(1, None, 0)
        root = TestNode(0, [b])

        expected = np.ones(d)
        self.assertTrue(np.allclose(q.calculate_embedding(root), expected))

#         a: [1,2,3]
#         b: [-1, 0, 1]

#         Wv * b = [3,0,2] 
#         Wv * a = [9,16,8]

#         hb = [1,-3,3]
#         ha = [7,13,9] + hb = [8,10,12]

#         hidden b: 0.5 b + 



# We = np.array([[1,2,3], [-1,0,1]])
# Wr = np.array([[
#     [1,2,3],
#     [2,3,1], 
#     [3,2,1]
# ]])
# Wv = np.array([
#     [-1, 2, 2],
#     [ 3, 2, 3],
#     [ 0, 1, 2]
# ])
# b = np.array([-2, -3, 1])

# xb = We[1]
# xa = We[0]

# hb = np.tanh(np.matmul(Wv,xb) + b)
# ha = np.tanh(np.matmul(Wv,xa) + np.matmul(Wr[0],xb) + b)