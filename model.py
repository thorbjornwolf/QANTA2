"""This module contains the QANTA MV-RNN model
"""

from multiprocess import Pool, cpu_count
import numpy as np
from time import time

import utils

class QANTA(object):
    """MV-RNN for DependencyTrees"""

    def __init__(self, dimensionality, vocabulary, dependency_dict,
                 learning_rate=0.05, nonlinearity=np.tanh, 
                 d_nonlinearity=utils.dtanh, embeddings_file=None):
        """dimensionality is a positive integer representing
            how many dimensions should go into word and relation 
            embeddings.
        vocabulary is a dict of words (strings) in the data, pointing 
            to index values for each word.
        dependency_dict is a dict of dependencies in the data, pointing to 
            index values for each dependency relation.
        nonlinearity is an elementwise nonlinear method to apply to numpy
            arrays. It is used in the calculation of hidden representations.
            Default is normalized np.tanh.
        d_nonlinearity is the differentiated nonlinearity
        embeddings_file is the path to a binary file in the word2vec format. 
            If defined, the model's vectors are initialized from that file.
        """
        self.dimensionality = dimensionality
        self.vocabulary = utils.Vocabulary(vocabulary)
        self.dependency_dict = utils.Vocabulary(dependency_dict)
        self.learning_rate = learning_rate
        self.answers = []

        self.nonlinearity = nonlinearity
        self.d_nonlinearity = d_nonlinearity

        self.generate_embeddings()
        if embeddings_file:
            self.load_embeddings_word2vec(embeddings_file)

        # TODO Consider how to do this in a more loopable, less verbose,
        # but still reader-friendly way
        self.adagrad_We = utils.Adagrad(learning_rate)
        self.adagrad_Wr = utils.Adagrad(learning_rate)
        self.adagrad_Wv = utils.Adagrad(learning_rate)
        self.adagrad_b = utils.Adagrad(learning_rate)

    def generate_embeddings(self, lo=-1, hi=1):
        """Generates We, Wr, Wb, and b and sets them in
        self.We, self.Wr, self.Wb, and self.b
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
        b = np.random.uniform(lo, hi, size=(d, 1))

        self.Wv = Wv
        self.b = b

    def load_embeddings_word2vec(self, path):
        """Loads word vectors from a binary file complying with the 
        word2vec format.
        """
        from gensim.models import Word2Vec
        # C binary format
        model = Word2Vec.load_word2vec_format(embeddings_file, binary=True)
        for word, index in vocabulary.iteritems():
            if word in model:
                self.We[index] = model[word]

    def train(self, trees, n_incorrect_answers=100, 
              n_epochs=30, n_batches=None):
        """Trains the QANTA model on the sentence trees.

        trees is a list of DependencyTree
        n_incorrect_answers is |Z| in the paper's eq. 5. It determines
            how many incorrect answers are sampled from the training data
            to be used in calculating sentence error.
        n_epochs is the number of times the model trains on the input data
        """
        # Grab new answers, and enforce uniqueness
        for t in trees:
            if not t.answer in self.answers:
                self.answers.append(t.answer)

        answers = self.answers # A tiny bit shorter
        
        # Make sure we can sample n_incorrect_answers different answers.
        if len(answers) - 1 < n_incorrect_answers:
            n_incorrect_answers = len(answers) - 1

        # QANTA original code says 'ideally 25 minibatches per epoch'
        n_batches = n_batches or min(25, len(trees))
        batch_size = len(trees) / n_batches

        if cpu_count() < batch_size:
            trainer = self._train_batch_parallelize
        else:
            # No reason to parallelize
            trainer = self._train_batch

        for epoch in xrange(n_epochs):
            epoch_error = 0
            epoch_start = time()
            for batch in xrange(n_batches):
                batch_start = time()

                # Index range for this batch
                lo = batch * batch_size
                hi = lo + batch_size

                batch_trees = trees[lo:hi]

                # Parallel training
                batch_error = trainer(batch_trees, n_incorrect_answers)
                # Serial training
                # batch_error = self._train_batch(batch_trees, n_incorrect_answers)

                # Only print batch stats if it takes more than 5 seconds
                if time() - batch_start > 5:
                    print ("Training error epoch {}, batch {}: {} "
                           "({:.2f} seconds)").format(epoch, batch, 
                           batch_error, time() - batch_start)
                epoch_error += batch_error

            if time() - epoch_start > 5:
                print ("Total training error for epoch {}: {} "
                       "({:.2f} seconds)").format(epoch, epoch_error, 
                        time() - epoch_start)

    def _train_batch_parallelize(self, trees, n_incorrect_answers):
        """Parallelizes training for a list of trees.
        Uses the number of threads given by multiprocessing.cpu_count()

        Updates model parameters directly, and returns batch error.
        """
        # Defaults to using cpu_count() threads
        pool = Pool()
        
        def get_subbatch_deltas(_trees):
            return self._train_batch(_trees, n_incorrect_answers, 
                                     apply_learning=False)

        subbatches = utils.split(trees, n_slices=cpu_count())

        # result will be a list of tuples (error, deltas)
        result = pool.map(get_subbatch_deltas, subbatches)

        # no more processes accepted by this pool
        pool.close()   
        # Wait until mapping is completed
        pool.join()

        error = sum([r[0] for r in result])
        deltas = [r[1] for r in result]
        for (delta_Wv, delta_b, delta_We, delta_Wr) in deltas:
            self.Wv -= delta_Wv
            self.b -=  delta_b
            self.We -= delta_We
            self.Wr -= delta_Wr

        return error

    def _train_batch(self, trees, n_incorrect_answers, shuffle=True, 
                     apply_learning=True):
        """Performs a single training run over the given trees.
        
        trees is a list of DependencyTree
        n_incorrect_answers is the number of answers used as 
            negative samples
        shuffle indicates whether or not to shuffle the tree list
        apply_learning indicates whether to apply the learned deltas
            directly to this model's parameters, or rather return them.
            If false, returns (batch error, deltas) where deltas are 
            (delta_Wv, delta_b, delta_We, delta_Wr)

        Returns batch error in accordance with eq. 6, or
        (batch error, deltas) if apply_learning is False
        """
        if shuffle:
            np.random.shuffle(trees)

        for tree in trees:
            # Sample from all other answers than the present one
            incorrect_answers = [x for x in self.answers if x != tree.answer]
            # randomize their order
            incorrect_answers = np.random.choice(incorrect_answers,
                                                 n_incorrect_answers, 
                                                 replace=False)
            # Calculate hidden representations and tree error
            self.forward_propagate(tree, incorrect_answers)
        
        error = sum([tree.error for tree in trees])

        # Eq. 6: Sum of error over all sentences, divided by number of nodes
        n_nodes = sum((t.n_nodes() for t in trees))
        se = error / n_nodes
        # se is a single value in a two-dimensional array
        try:
            se = se[0][0]
        except TypeError:
            # But it is not always for some reason!
            pass

        # Initialize deltas
        d = self.dimensionality
        delta_Wv = np.zeros((d, d))
        delta_b = np.zeros((d, 1))
        delta_We = np.zeros((len(self.vocabulary), d))
        delta_Wr = np.zeros((len(self.dependency_dict), d, d))

        deltas = (delta_Wv, delta_b, delta_We, delta_Wr)

        # Backpropagation
        for tree in trees:
            self.back_propagate(tree, *deltas)

        # Scale deltas
        for d in deltas:
            d /= n_nodes

        self.adagrad_scale(*deltas) # delta_Wv, delta_b, delta_We, delta_Wr

        # Apply learning
        if apply_learning:
            self.Wv -= delta_Wv
            self.b -=  delta_b
            self.We -= delta_We
            self.Wr -= delta_Wr
            
            return se
        return se, deltas

    def forward_propagate(self, tree, wrong_answers):
        """Calculates tree-wide error and node-wise answer_delta, as well
        as node.hidden (the node's hidden representation) and 
        node.hidden_norm, the normalized version of the node's hidden 
        representation.

        tree is a DependencyTree
        wrong_answers is a list of strings of answers excluding the correct
            answer to the sentence posed in tree

        This method sets
            tree.error
            node.answer_delta
            node.hidden
            node.hidden_norm

        This is reminiscent of eq. 5, although L and rank are not used.

        Does not return anything, but modifies tree and its nodes directly.
        """

        self.set_hidden_representations(tree)

        tree.error = 0.0
        tree_answer_We = self.word2embedding(tree.answer)

        for node in tree.iter_nodes():
            node.answer_delta = np.zeros((self.dimensionality, 1))

            answer_similarity = tree_answer_We.T.dot(node.hidden_norm)
            base_error = 1 - answer_similarity

            for z in wrong_answers:
                # d,1
                z_We = self.word2embedding(z)
                # d,1
                similarity = z_We.T.dot(node.hidden_norm)
                z_error = base_error + similarity
                if z_error > 0:
                    tree.error += z_error
                    node.answer_delta += z_We - tree_answer_We

    def back_propagate(self, tree, delta_Wv, delta_b, delta_We, delta_Wr):
        """Backpropagation in the same manner as seen in the original
        QANTA implementation.

        tree is a DependencyTree that has been processed by forward_propagate.

        Assumes each node has 
            node.answer_delta
            node.hidden, its hidden representation seen in eq. 4, 
            node.hidden_norm, its normalized node.hidden vector.

        Modifies delta_Wv, delta_b, delta_We, delta_Wr
        Returns nothing
        """
        
        deltas = dict()

        # Iterate over tree nodes breadth-first, top to bottom
        for node in tree.iter_nodes_from_root():
            
            if node == tree.root:
                # shape d,1
                act = node.answer_delta
            else:
                # shape d,1
                parent_delta = deltas[node.parent]
                act = (self.dep2embedding(node.dependency).dot(parent_delta) + 
                        node.answer_delta)
            
                # grads[0][rel]
                node_Wr_index = self.dependency2index(node.dependency)
                # shape d,d
                delta_Wr[node_Wr_index] += parent_delta.dot(node.hidden_norm.T)

            # vector of d elements
            node_delta = self.d_nonlinearity(node.hidden).dot(act)

            # grads[1]   d,d
            delta_Wv += node_delta.dot(self.word2embedding(node.word).T)
            # grads[2] d
            delta_b += node_delta
            # grads[3][:, word embedding index] 1,d
            index = self.word2index(node.word)
            # [(index,),] ensures that the slice has shape 1,d
            # and the .T at the end ensures the input has shape 1,d
            delta_We[(index,),] += (self.Wv.dot(node_delta)).T

            deltas[node] = node_delta

    def predict(self, tree, n_best=None):
        """Predicts the answer to a question stated in tree.

        tree is a DependencyTree
        n_best is how many of the closest answers that should 
            be returned. If undefined, returns the single best
            answer as a string. If defined, returns a list of
            the n_best answers as strings.

        Returns a string or a list, depending on n_best
        """
        self.set_hidden_representations(tree)
        # Get sentence representation, L2 normalize it
        pred_vector = np.sum([n.hidden_norm for n in tree.iter_nodes()], axis=0)
        pred_vector = utils.normalize(pred_vector)

        # Get answer representations, L2 normalize them
        ans_idx = map(self.word2index, self.answers)
        candidates = self.We[ans_idx]
        
        norm_factor = np.linalg.norm(candidates, axis=1)[:,np.newaxis]
        norm_candidates = candidates / norm_factor

        # Find closest vector using dot product
        dotted = norm_candidates.dot(pred_vector)
        if n_best is None:
            best_match_idx = np.argmax(dotted)
            return self.answers[best_match_idx]
        # Get the indices of the n highest dot values
        best_idx = dotted.flatten().argsort()[-n_best:]
        # Reverse their order, as they are lowest to highest
        best_idx = best_idx[::-1]

        return [self.answers[i] for i in best_idx]

    def predict_many(self, trees, n_best=None):
        return [self.predict(t, n_best) for t in trees]

    def get_accuracy(self, trees, within_n_best=1):
        """Returns the accuracy for this model on the given
        data.

        trees is a list of DependencyTree
        within_n_best is a positive integer. If the correct answer is
            not within the within_n_best closest answers to the tree's 
            representation, it counts as a miss. Otherwise a hit.
        """
        pred = self.predict_many(trees, n_best=within_n_best)
        truth = [t.answer for t in trees]
        n_correct = 0
        for p,t in zip(pred, truth):
            if t in p: 
                n_correct += 1

        return float(n_correct) / len(pred)

    def set_hidden_representations(self, tree):
        """For each node in the tree, calculates the hidden representation
        and stores it in node.hidden. Additionally calculates the normalized
        hidden representation and stores it in node.hidden_norm.

        This is the paper's equation 4.

        Does not return anything, but modifies tree and nodes directly.
        """
        # Start with bottom-most leaf nodes, then the layer above,
        # et cetera up to the root
        for node in reversed(tree.iter_nodes_from_root()):
            children_sum = None
            for c in node.children:
                val = self.dep2embedding(c.dependency).dot(c.hidden_norm)
                assert val.shape == (self.dimensionality,1) 
                if children_sum is None:
                    children_sum = val
                else:
                    children_sum += val
            
            if children_sum is None:
                children_sum = 0
            else:
                assert children_sum.shape == (self.dimensionality,1) 

            f = self.nonlinearity
            word_hidden = self.Wv.dot(self.word2embedding(node.word))
            node.hidden = f(word_hidden + children_sum + self.b)
            node.hidden_norm = utils.normalize(node.hidden)

    #### Utility methods ####

    def word2embedding(self, word):
        """Returns word embedding from self.We
        as a d,1 numpy array
        """
        index = self.vocabulary[word]
        values = self.We[index]
        vector = np.expand_dims(values, axis=1)
        return vector

    def word2index(self, word):
        return self.vocabulary[word]

    def dep2embedding(self, dependency):
        """Returns dependency embedding from self.Wr
        as a d,d numpy array
        """
        index = self.dependency_dict[dependency]
        matrix = self.Wr[index]
        return matrix

    def dependency2index(self, dependency):
        return self.dependency_dict[dependency]

    def adagrad_scale(self, delta_Wv, delta_b, delta_We, delta_Wr):
        delta_Wv *= self.adagrad_Wv.get_scale(delta_Wv)
        delta_b  *= self.adagrad_b.get_scale(delta_b)
        delta_We *= self.adagrad_We.get_scale(delta_We)
        delta_Wr *= self.adagrad_Wr.get_scale(delta_Wr)
