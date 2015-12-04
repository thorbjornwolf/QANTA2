# Run with `python -m unittest test.test_model`

import unittest

import numpy as np

import model

class TestNode(object):
    def __init__(self, word_index, children=None, dependency_index=None):
        self.word_index = word_index
        self.children = children or []
        self.dependency_index = dependency_index

    def n_nodes(self):
        s = sum([n.n_nodes() for n in self.children])
        return s + 1

    def iter_nodes(self, nodes=None):
        if nodes is None:
            nodes = []
        nodes.append(self)
        for child in self.children:
            child.iter_nodes(nodes)
        return nodes

class TestTree(object):
    def __init__(self, root, answer_index):
        self.root = root
        self.answer_index = answer_index

    def n_nodes(self):
        return self.root.n_nodes()

    def iter_nodes(self):
        return self.root.iter_nodes()
        

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
        self.assertEquals(q.b.shape, (5,))

    def test_representation_calculation(self):
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
        self.assertTrue(np.allclose(q.get_node_representation(root), expected))

    def test_train_breaks(self):
        vocab = {'alpha':0, 'bravo':1, '42':2, 'not 42':3, 'also wrong':4}
        deplist = {'prep':0}
        d = 3
        answers = {2, 3, 4}

        b = TestNode(1, None, 0)
        root = TestNode(0, [b])

        tt = TestTree(root, 0)

        q = model.QANTA(d, vocab, deplist, answers)
        q.train([tt])
