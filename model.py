"""This module contains the QANTA MV-RNN model
"""

import numpy as np

class QANTA(object):
    """MV-RNN for DependencyTrees"""

    def __init__(self, dimensionality, word_vocabulary, dependency_list,
                 answers, nonlinearity=np.tanh, 
                 load_embeddings_from_file=False):
        """dimensionality is a positive integer representing
            how many dimensions should go into word and relation 
            embeddings.
        word_vocabulary is a dict of words (strings) in the data, pointing 
            to index values for each word.
        dependency_list is a dict of dependencies in the data, pointing to 
            index values for each dependency relation.
        answers is a list or set of the possible answers in the data
        nonlinearity is an elementwise nonlinear method to apply to numpy
            arrays. It is used in the calculation of hidden representations.
            Default is np.tanh.
        """
        self.dimensionality = dimensionality
        self.word_vocabulary = word_vocabulary
        self.dependency_list = dependency_list
        self.answers = answers
        self.nonlinearity = nonlinearity

        if load_embeddings_from_file:
            raise NotImplementedError()
        else:
            self.generate_embeddings()


    def generate_embeddings(self, lo=-1, hi=1):
        """Generates We, Wr, Wb, and b
        """
        d = self.dimensionality

        # We: Word embeddings
        # #words x d
        # word embeddings is found by `We[vocab['myword']]`
        We = np.random.uniform(lo, hi, size=(len(self.word_vocabulary), d))

        # Wr: Relation embedded matrices (original paper called it Wr)
        # #dependencies x d x d
        Wr = np.random.uniform(lo, hi, size=(len(self.dependency_list), d, d))

        self.We = We
        self.Wr = Wr

        # Wv: Matrix that transforms embeddings to hidden representation
        Wv = np.random.uniform(lo, hi, size=(d, d))
        # b: bias vector added in each transformation to hidden representation
        b = np.random.uniform(lo, hi, size=(d, 1))

        self.Wv = Wv
        self.b = b

    def predict(self, dependency_tree):
        h = self.calculate_embedding(dependency_tree)
        # Find closest match among answers, return that

    def calculate_embedding(self, node):
        """node is a dependency tree node, or a dependency tree.

        Returns the node's recursive hidden value, or the hidden
            value of the tree's root node

        This is eq. 4 in the QANTA paper.
        """

        # Is node actually a tree?
        # TODO use proper type checking when the types get implemented.
        if 'root' in dir(node):
            node = node.root

        # Calculate sum of relation-modified children 
        # representations 
        # In paper eq. 4: sigma_{k in K}(W_{R(n,k)} * h_k)
        children_sum = np.zeros(self.dimensionality)
        for i, child in enumerate(node.children):
            hc = self.calculate_embedding(child)
            Wc = self.Wr[child.dependency_index]
            Wrh = np.matmul(Wc, hc)

            children_sum += Wrh


        x = self.We[node.word_index]
        node_h_arg = np.matmul(self.Wv, x) + children_sum + self.b
        node_h = self.nonlinearity(node_h_arg)
        return node_h

