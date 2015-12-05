"""This module contains the QANTA MV-RNN model
"""

import numpy as np

class QANTA(object):
    """MV-RNN for DependencyTrees"""

    def __init__(self, dimensionality, vocabulary, dependency_dict,
                 answers, nonlinearity=None,
                 embeddings_file=None):
        """dimensionality is a positive integer representing
            how many dimensions should go into word and relation 
            embeddings.
        vocabulary is a dict of words (strings) in the data, pointing 
            to index values for each word.
        dependency_dict is a dict of dependencies in the data, pointing to 
            index values for each dependency relation.
        answers is a list of the possible answers in the data
        nonlinearity is an elementwise nonlinear method to apply to numpy
            arrays. It is used in the calculation of hidden representations.
            Default is normalized np.tanh.
        embeddings_file is the path to a binary file in the word2vec format. 
            If defined, the model's vectors are initialized from that file.
        """
        self.dimensionality = dimensionality
        self.vocabulary = vocabulary
        self.dependency_dict = dependency_dict
        self.answers = answers

        if nonlinearity is None:
            nonlinearity = self.ntanh
        self.nonlinearity = nonlinearity

        self.generate_embeddings()
        if embeddings_file:
            self.load_embeddings_word2vec(embeddings_file)

    def word2index(self, word):
        return self.vocabulary[word]

    def dependency2index(self, dependency):
        return self.dependency_dict[dependency]

    def ntanh(x):
        """Normalized tanh"""
        tanh = np.tanh(x)
        return tanh / np.linalg.norm(tanh)

    def generate_embeddings(self, lo=-1, hi=1):
        """Generates We, Wr, Wb, and b
        """
        d = self.dimensionality

        # We: Word embeddings
        # #words x d
        # word embeddings is found by `We[vocab['myword']]`
        We = np.random.uniform(lo, hi, size=(len(self.vocabulary), d))

        # Wr: Relation embedded matrices (original paper called it Wr)
        # #dependencies x d x d
        Wr = np.random.uniform(lo, hi, size=(len(self.dependency_dict), d, d))

        self.We = We
        self.Wr = Wr

        # Wv: Matrix that transforms embeddings to hidden representation
        Wv = np.random.uniform(lo, hi, size=(d, d))
        # b: bias vector added in each transformation to hidden representation
        b = np.random.uniform(lo, hi, size=(d))

        self.Wv = Wv
        self.b = b

    def load_embeddings_word2vec(self, path):
        """Loads vectors from a binary file complying with the 
        word2vec format.
        """
        from gensim.models import Word2Vec
            # C binary format
            model = Word2Vec.load_word2vec_format(embeddings_file, binary=True)
            for word, index in vocabulary.iteritems():
                if word in model:
                    self.We[index] = model[word]

    def train(self, dependency_trees, n_incorrect_answers=100):
        """Trains the QANTA model on the sentence trees.

        n_incorrect_answers is |Z| in the paper's eq. 5. It determines
            how many incorrect answers are sampled from the training data
            to be used in calculating sentence error.
        """

        if len(self.answers) - 1 < n_incorrect_answers:
            print ("Cannot sample without replacement from {} answers, as "
                   "only {} answers are available. Setting "
                   "n_incorrect_answers down to {}.").format(
                   n_incorrect_answers, len(self.answers), len(self.answers) - 1)
            
            n_incorrect_answers = len(self.answers) - 1

        error = 0
        for tree in dependency_trees:

            # Sample from all other answers than the present one
            incorrect_answers = [x for x in self.answers if x != tree.answer]

            incorrect_answers = np.random.choice(incorrect_answers,
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
            Wc = self.Wr[self.dependency2index(child.dependency)]
            Wrh = np.matmul(Wc, hc)

            children_sum += Wrh

        x = self.We[self.word2index(node.word)]
        node_h_arg = np.matmul(self.Wv, x) + children_sum + self.b
        node_h = self.nonlinearity(node_h_arg)
        return node_h

    def node_error(self, node, answer, incorrect_answers):
        """node is a DependencyNode (s in paper eq. 5)
        answer is a string (c in paper eq. 5)
        incorrect_answers is a list of strings (Z in paper eq. 5)

        Returns the inner sum over Z in eq. 5
        """
        error_per_incorrect_answer = []

        hs = self.get_node_representation(node)
        xc = self.We[self.word2index(answer)]
        xc_dot_hs = np.dot(xc, hs)

        for z in incorrect_answers:
            # Calculate max_term: max(0, 1 - x_c*h_s + x_z*h_s)

            xz = self.We[self.word2index(z)]
            max_term = 1 - xc_dot_hs + np.dot(xz, hs)
            if max_term <= 0:
                # No need to do the ranking calculation
                continue

            # Calcuate L_term: L(rank(c,s,Z))
            # To do that, calculate rank_approx
            # NB: Original implementation did not use this!
            rank_approx = self.approximate_rank(xc, hs, incorrect_answers, xc_dot_hs)

            L_term = np.sum(1. / np.arange(1, rank_approx))

            error_per_incorrect_answer.append(L_term * max_term)

        return sum(error_per_incorrect_answer)

    def approximate_rank(self, xc, hs, Z, xc_dot_hs=None):
        """Calculates the rank approximation for 
        sentence errors (paper eq. 5)

        xc is the word embedding for the correct answer
        hs is the hidden state of the node being evaluated
        Z is the list of strings of wrong answers
        xc_dot_hs is an optional precomputation of xc dot hs
        """
        # Randomly permutate the incorrect answers
        permutation = np.random.permutation(Z)
        # Create a generator object for iterating over the incorrect
        # answers' embeddings
        incorrect_embeddings = (self.We[self.word2index(w)] for w in permutation)

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
        incorrect_answers is a list of string that are not the answer to
            the sentence in sentence_tree
        """

        # TODO: Reuse node hidden vector calculations

        node_error = []
        for node in sentence_tree.iter_nodes():
            e = self.node_error(node, sentence_tree.answer, incorrect_answers)
            node_error.append(e)

        return sum(node_error)