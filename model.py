"""This module contains the QANTA MV-RNN model
"""

from __future__ import print_function

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
        b = np.random.uniform(lo, hi, size=(d))

        self.Wv = Wv
        self.b = b

    def train(self, dependency_trees, n_incorrect_answers=100):
        """Trains the QANTA model on the sentence trees.

        n_incorrect_answers is |Z| in the paper's eq. 5. It determines
            how many incorrect answers are sampled from the training data
            to be used in calculating sentence error.
        """

        if len(self.answers) - 1 < n_incorrect_answers:
            print(("Cannot sample without replacement from {} answers, as "
                   "only {} answers are available. Setting "
                   "n_incorrect_answers down to {}.").format(
                   n_incorrect_answers, len(self.answers), len(self.answers) - 1))
            
            n_incorrect_answers = len(self.answers) - 1

        error = 0
        for tree in dependency_trees:

            # Sample from all other answers than the present one
            incorrect_answers = self.answers.difference([tree.answer_index])

            incorrect_answers = np.random.choice(list(incorrect_answers),
                                                 n_incorrect_answers, 
                                                 replace=False)
            
            err = self.sentence_error(tree, incorrect_answers)
            error += err

        # Eq. 6: Sum of error over all sentences, divided by number of nodes
        n_nodes = sum((t.n_nodes() for t in dependency_trees))
        se = error / n_nodes

        # TODO Update weights!

    def predict(self, dependency_trees):
        h = self.get_paragraph_representation(dependency_trees)
        # TODO: Find closest match among answers, return that

    def get_paragraph_representation(self, trees):
        """trees is a list of DependencyTree, presumed to be part
        of a larger consecutive paragraph.

        As is unique to QANTA, this representation is simply the average
            of the contained sentence representations.
        """

        return np.average([self.get_sentence_representation(t) for t in trees],
                          axis=0)

    def get_sentence_representation(self, tree):
        return self.get_node_representation(tree.root)

    def get_node_representation(self, node):
        """node is a dependency tree node, or a dependency tree.

        Returns the node's recursive hidden value, or the hidden
            value of the tree's root node

        This is eq. 4 in the QANTA paper.
        """

        # Calculate sum of relation-modified children
        # representations
        # In paper eq. 4: sigma_{k in K}(W_{R(n,k)} * h_k)
        children_sum = np.zeros(self.dimensionality)
        for i, child in enumerate(node.children):
            hc = self.get_node_representation(child)
            Wc = self.Wr[child.dependency_index]
            Wrh = np.matmul(Wc, hc)

            children_sum += Wrh

        x = self.We[node.word_index]
        node_h_arg = np.matmul(self.Wv, x) + children_sum + self.b
        node_h = self.nonlinearity(node_h_arg)
        return node_h

    def node_error(self, node, answer, incorrect_answers):
        """node is a DependencyNode (s in paper eq. 5)
        answer is a vocabulary index (c in paper eq. 5)
        incorrect_answers is a listlike of vocabulary indices
            (Z in paper eq. 5)

        Returns the inner sum over Z in eq. 5
        """
        error_per_incorrect_answer = []

        hs = self.get_node_representation(node)
        xc = self.We[answer]
        xc_dot_hs = np.dot(xc, hs)

        # 1 - x_c*h_s is reused many times
        one_minus_xc_dot_hs = 1 - xc_dot_hs

        for z in incorrect_answers:
            # Calculate max_term: max(0, 1 - x_c*h_s + x_z*h_s)
            xz = self.We[z]
            max_term = 1 - xc_dot_hs + np.dot(xz, hs)
            if max_term <= 0:
                # No need to do the ranking calculation
                continue

            # Calcuate L_term: L(rank(c,s,Z))
            # To do that, calculate rank_approx
            # NB: Original implementation did not use this!
            # NB: We're rounding the value up: It must be int, and >= 1
            rank_approx = np.ceil(self.approximate_rank(xc, hs,
                                                        incorrect_answers, xc_dot_hs))

            L_term = np.sum(1. / (1 + np.arange(rank_approx)))

            error_per_incorrect_answer.append(L_term * max_term)

        return sum(error_per_incorrect_answer)

    def approximate_rank(self, xc, hs, Z, xc_dot_hs=None):
        """Calculates the rank approximation for 
        sentence errors (paper eq. 5)

        xc is the word embedding for the correct answer
        hs is the hidden state of the node being evaluated
        Z is the list of indices of wrong answers
        xc_dot_hs is an optional precomputation of xc dot hs
        """
        # Randomly permutate the incorrect answers
        permutation = np.random.permutation(Z)
        # Create a generator object for iterating over the incorrect
        # answers' embeddings
        incorrect_embeddings = (self.We[i] for i in permutation)

        if xc_dot_hs is None:
            xc_dot_hs = np.dot(xc, hs)

        # Get the count k for number of random samples from the incorrect
        # embeddings needed to find a violation of the wanted margin
        for k, xz in enumerate(incorrect_embeddings):
            if xc_dot_hs < 1 + np.dot(xz, hs):
                break

        if k == 0:
            return 1
        rank = (len(Z) - 1) / float(k)
        return rank

    def sentence_error(self, sentence_tree, incorrect_answers):
        """Calculates the error for a sentence, using the incorrect
        answers as contrast as per eq. 5.

        In the QANTA paper, this _is_ equation 5

        sentence_tree is a DependencyTree
        incorrect_answers is a listlike of indices from the vocabulary
        """

        # TODO: Reuse node hidden vector calculations

        node_error = []
        for node in sentence_tree.iter_nodes():
            node_error.append(self.node_error(node, sentence_tree.answer_index,
                              incorrect_answers))

        return sum(node_error)