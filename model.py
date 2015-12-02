"""This module contains the QANTA MV-RNN model
"""

import numpy as np

class QANTA(object):
    """MV-RNN for DependencyTrees"""

    def __init__(self, dimensionality, word_vocabulary, dependency_list,
                 load_embeddings_from_file=False):
        """dimensionality is a positive integer representing
            how many dimensions should go into word and relation 
            embeddings.
        word_vocabulary is a dict of words (strings) in the data, pointing 
            to index values for each word.
        dependency_list is a dict of dependencies in the data, pointing to 
            index values for each dependency relation.
        """
        self.dimensionality = dimensionality
        self.word_vocabulary = word_vocabulary
        self.dependency_list = dependency_list

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
