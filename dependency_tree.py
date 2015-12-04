from Queue import Queue


class DependencyNode(object):

    def __init__(self, word_index, sentence_index, dependency_index, children=None):
        """
        word_index is the index of the word in the global vocabulary
        sentence_index is the index of the word in the source sentence
        dependency_index is the index of the parent's dependency 
            relation to this node
        """
        self.word_index = word_index
        self.sentence_index = sentence_index
        self.dependency_index = dependency_index
        self.children = children or []

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

    def __repr__(self, prepend=''):
        s = "{}DependencyNode wi:{}, si:{}, di:{}".format(prepend, 
            self.word_index, self.sentence_index, self.dependency_index)
        output = [s]
        for c in sorted(self.children, key=lambda x: len(x.children)):
            output.append(c.__repr__(prepend+'    '))
        return '\n'.join(output)

class DependencyTree(object):
    """Represents a sentence with dependency relations

    Internal representation uses only indices, not actual words
    """

    def __init__(self, question_id):
        self.question_id = question_id

    def add(self, node, parent_sentence_index=None):
        """Inserts a node into the tree structure. 
        Nodes are inserted breadt first, to accomodate the style of the
            Stanford Parser output.
        If parent_sentence_index is None, it is assumed that the node is
            the root node.
        """
        if parent_sentence_index is None:
            self.root = node
            return
        parent = self.find_node_by_sentence_index(parent_sentence_index)
        assert parent is not None, "Cannot add node to tree; no parent found for it."

        parent.children.append(node)

    def find_node_by_sentence_index(self, sentence_index):
        for node in self.iter_nodes():
            if node.sentence_index == sentence_index:
                return node
        return None

    def n_nodes(self):
        return self.root.n_nodes()

    def iter_nodes(self):
        return self.root.iter_nodes()

    def __repr__(self):
        s = "DependencyTree qid:{}, n_nodes:{}".format(self.question_id, 
                self.n_nodes())
        return '\n'.join([s, self.root.__repr__('    ')])

    def get_ordered_words(self, vocabulary):
        rev_vocab = dict(((v,k) for k,v in vocabulary.iteritems()))
        words = []
        for node in sorted(self.iter_nodes(), key=lambda x: x.sentence_index):
            words.append(rev_vocab[node.word_index])
        return words


def trees_from_stanford_parse_tuples(list_of_stanford_parse_tuples, 
                                answer_indices, vocabulary, dependency_dict):
    """Takes a list of lists of tuples that are output from the Stanford
    Parser, and returns a list of DependencyTree. Order is preserved.

    list_of_stanford_parse_tuples is a list of lists of tuples following the 
        Stanford Parser output format, so that each element in the outer 
        list contains a sentence representation.
    answer_indices is a list of integers. Each integer is the index in the 
        vocabulary to the correct answer to the corresponding sentence in
        list_of_stanford_parse_tuples
    vocabulary is a dict where word indices can be looked up. The index of
        a word can then be used to find its embedding in the word embedding
        matrix We (used elsewhere in QANTA)
    dependency_dict is like vocabulary, but for the set of words representing
        the different kinds of relations found in the data.

    Returns a list of DependencyTree.
    """
    return [tree_from_stanford_parse_tuples(s, ai, vocabulary, dependency_dict) for s, ai 
            in zip(list_of_stanford_parse_tuples, answer_indices)]


def tree_from_stanford_parse_tuples(stanford_parse_tuples, answer_index, 
                                    vocabulary, dependency_dict):
    """Takes a list of tuples as output from the Stanford Parser,
    and returns a dependency tree for the sentence.

    stanford_parse_tuples is a list of tuples following the Stanford Parser 
        output format
    answer_index is a single integer, namely the index of the correct answer
        to the sentence encoded in stanford_parse_tuples, in the vocabulary
    vocabulary is a dict where word indices can be looked up. The index of
        a word can then be used to find its embedding in the word embedding
        matrix We (used elsewhere in QANTA)
    dependency_dict is like vocabulary, but for the set of words representing
        the different kinds of relations found in the data.

    Returns a single DependencyTree.
    """

    assert stanford_parse_tuples[0][2][0][0] == 'root', "No root in tree, cannot compute!"

    root_index = stanford_parse_tuples[0][2][0][1][0]

    queue = Queue()
    # Store items in queue as (SP index, relation index) pairs
    queue.put((root_index, None, None))

    tree = DependencyTree(answer_index)

    while not queue.empty():
        sentence_index, dependency_index, parent_sentence_index = queue.get()

        # Word indices and stanford_parse_tuples list indices are not
        # guaranteed to correspond, so find tuple by index by explicitly
        # looking it up
        for tup in stanford_parse_tuples:
            if tup[0] == sentence_index:
                # Stop searching for a match
                break

        word_index = vocabulary[tup[1]]

        # We explicitly pass None as node children, as they will be added
        # in a later iteration.
        node = DependencyNode(word_index, sentence_index, dependency_index, None)    
        tree.add(node, parent_sentence_index)

        children = tup[2]

        for child in children:
            depindex = dependency_dict[child[0]]
            children_indices = child[1]

            # There might possibly be more than one child stored under the
            # same relation, so iterate over them
            for ci in children_indices:
                queue.put((ci, depindex, sentence_index))

    return tree