# Run with `python -m unittest test.test_model`

import unittest

import numpy as np

import model
from dependency_tree import DependencyTree, DependencyNode

class TestNode(object):
    def __init__(self, word, children=None, dependency=None):
        self.word = word
        self.children = children or []
        self.dependency = dependency

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
    def __init__(self, root, answer):
        self.root = root
        self.answer = answer

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

        q = model.QANTA(d, vocab, deplist)

        # Word embeddings + 1 for unknown words
        self.assertEquals(q.We.shape, (3+1, 5)) 
        # Relation embeddings + 1 for unknown relations
        self.assertEquals(q.Wr.shape, (2+1, 5, 5))
        self.assertEquals(q.Wv.shape, (5, 5))
        self.assertEquals(q.b.shape, (5,1))

    # def test_representation_calculation(self):
    #     vocab = {'alpha':0, 'bravo':1}
    #     deplist = {'prep':0}
    #     d = 3
    #     answers = None

    #     q = model.QANTA(d, vocab, deplist)
    #     q.We = np.array([[1,2,3], [-1,0,1]])
    #     q.Wr = np.array([[
    #         [1,2,3],
    #         [2,3,1], 
    #         [3,2,1]
    #     ]])
    #     q.Wv = np.array([
    #         [-1, 2, 2],
    #         [ 3, 2, 3],
    #         [ 0, 1, 2]
    #     ])
    #     q.b = np.array([-2, -3, 1])


    #     b = TestNode('bravo', None, 'prep')
    #     root = TestNode('alpha', [b])

    #     expected = np.ones(d)
    #     self.assertTrue(np.allclose(q.get_node_representation(root), expected))

class TestTrain(unittest.TestCase):

    def test_train_runs(self):
        # "bravo 42 also wrong alpha"
        #    1    2   3    4     5
        # root: also
        #       dep0: bravo
        #           dep1: 42
        #       dep1: wrong
        #           dep0: alpha
        vocab = {'alpha':0, 'bravo':1, '42':2, 'not 42':3, 'also': 4, 'wrong':5}
        deplist = {'dep0':0, 'dep1':1}
        d = 3
        answers = ['42', 'not 42']

        tree = DependencyTree('42')
        tree.add(DependencyNode('also', 3, None)) # root
        tree.add(DependencyNode('bravo', 1, 'dep0'), 3) 
        tree.add(DependencyNode('42', 2, 'dep1'), 1) 
        tree.add(DependencyNode('wrong', 4, 'dep1'), 3) 
        tree.add(DependencyNode('alpha', 5, 'dep0'), 4) 

        root2 = DependencyNode('not 42', 1, None, [])
        tree2 = DependencyTree('not 42')
        tree2.add(root2)

        q = model.QANTA(d, vocab, deplist)
        q.train([tree, tree2])


def extract_vocab_and_deps(trees):
    vocab = dict()
    deps = dict()

    def to_voc(word):
        if not word in vocab:
            vocab[word] = len(vocab)

    def to_dep(dep):
        if not dep in deps:
            deps[dep] = len(deps)

    for tree in trees:
        to_voc(tree.answer)
        for n in tree.iter_nodes():
            to_voc(n.word)
            to_dep(n.dependency)

    return vocab, deps


class TestPredict(unittest.TestCase):
    def __init__(self, arg):
        super(TestPredict, self).__init__(arg)
        self.trees = []
        tree = DependencyTree('Margaret')

        tree.add(DependencyNode('is',2, 'b'), None)
        tree.add(DependencyNode('She', 1, 'a'), 2)
        tree.add(DependencyNode('Denmark\'s', 3, 'c'), 2)
        tree.add(DependencyNode('queen', 4, 'd'), 3)
        self.trees.append(tree)

        tree = DependencyTree('BMW')

        tree.add(DependencyNode('car', 3, 'b'), None)
        tree.add(DependencyNode('famous',2, 'c'), 3)
        tree.add(DependencyNode('brand', 4, 'd'), 3)
        tree.add(DependencyNode('A', 1, 'a'), 2)
        self.trees.append(tree)

        tree = DependencyTree('Horse')

        tree.add(DependencyNode('large',2, 'c'), None)
        tree.add(DependencyNode('A', 1, 'd'), 2)
        tree.add(DependencyNode('scary', 3, 'b'), 2)
        tree.add(DependencyNode('animal', 4, 'a'), 3)
        self.trees.append(tree)

    def test_predict_2_classes(self):
        self.generic_test(d=2, n_trees=2, 
                          runs=50, n_epochs=100, 
                          min_acc=0.9)

    def test_predict_3_classes(self):
        self.generic_test(d=2, n_trees=3, 
                          runs=50, n_epochs=100, 
                          min_acc=0.85)

    def generic_test(self, d, n_trees, runs, n_epochs, min_acc):
        trees = self.trees[:n_trees]
        vocab, deplist = extract_vocab_and_deps(trees)
        q = model.QANTA(d, vocab, deplist)

        total_acc = 0
        for _ in range(runs):
            q.train(trees, n_epochs=n_epochs)
            total_acc += q.get_accuracy(trees,1)
  
        acc = total_acc / float(runs)
        self.assertGreater(acc, min_acc)