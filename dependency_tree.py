from Queue import Queue


class DependencyNode(object):

    def __init__(self, word, index_in_sentence, dependency, children=None):
        """
        word is the word that the node represents
        index_in_sentence is the index of the word in the source sentence.
            Note that this starts at 1, not 0
        dependency is a string representing the parent's dependency relation
            to this node
        """
        self.word = word
        self.index_in_sentence = index_in_sentence
        self.dependency = dependency
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
            self.word, self.index_in_sentence, self.dependency)
        output = [s]
        for c in sorted(self.children, key=lambda x: len(x.children)):
            output.append(c.__repr__(prepend+'    '))
        return '\n'.join(output)

class DependencyTree(object):
    """Represents a sentence with dependency relations

    Internal representation uses only indices, not actual words
    """

    def __init__(self, answer):
        """answer is a string representing the correct answer
        to the question posed by the sentence represented by this tree.
        """
        self.answer = answer

    def add(self, node, parent_index_in_sentence=None):
        """Inserts a node into the tree structure. 
        Nodes are inserted breadth first, to accomodate the style of the
            Stanford Parser output.
        If parent_index_in_sentence is None, it is assumed that the node is
            the root node.
        """
        if parent_index_in_sentence is None:
            self.root = node
            return
        parent = self.find_node_by_index_in_sentence(parent_index_in_sentence)
        assert parent is not None, "Cannot add node to tree; no parent found for it."

        parent.children.append(node)

    def find_node_by_index_in_sentence(self, index_in_sentence):
        for node in self.iter_nodes():
            if node.index_in_sentence == index_in_sentence:
                return node
        return None

    def n_nodes(self):
        return self.root.n_nodes()

    def iter_nodes(self):
        return self.root.iter_nodes()

    def __repr__(self):
        s = "DependencyTree ai:{}, n_nodes:{}".format(self.answer, self.n_nodes())
        return '\n'.join([s, self.root.__repr__('    ')])

    def get_ordered_words(self, vocabulary):
        words = []
        for node in sorted(self.iter_nodes(), key=lambda x: x.index_in_sentence):
            words.append(node.word)
        return words


def trees_from_stanford_parse_tuples(list_of_stanford_parse_tuples, 
                                answers, vocabulary, dependency_dict):
    """Takes a list of lists of tuples that are output from the Stanford
    Parser, and returns a list of DependencyTree. Order is preserved.

    list_of_stanford_parse_tuples is a list of lists of tuples following the 
        Stanford Parser output format, so that each element in the outer 
        list contains a sentence representation.
    answers is a list of strings. Each string is the correct answer to the
        corresponding sentence in list_of_stanford_parse_tuples
    vocabulary is a dict where word indices can be looked up. The index of
        a word can then be used to find its embedding in the word embedding
        matrix We (used elsewhere in QANTA)
    dependency_dict is like vocabulary, but for the set of words representing
        the different kinds of relations found in the data.

    Returns a list of DependencyTree.
    """
    output = []
    for sptuple, answer in zip(list_of_stanford_parse_tuples, answers):
        output.append(tree_from_stanford_parse_tuples(sptuple, answer, 
                                                vocabulary, dependency_dict))


def tree_from_stanford_parse_tuples(stanford_parse_tuples, answer, 
                                    vocabulary, dependency_dict):
    """Takes a list of tuples as output from the Stanford Parser,
    and returns a dependency tree for the sentence.

    stanford_parse_tuples is a list of tuples following the Stanford Parser 
        output format
    answer is a single string, namely the correct answer to the sentence 
        encoded in stanford_parse_tuples
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

    tree = DependencyTree(answer)

    while not queue.empty():
        index_in_sentence, dependency, parent_index_in_sentence = queue.get()

        # Word indices and stanford_parse_tuples list indices are not
        # guaranteed to correspond, so find tuple by index by explicitly
        # looking it up
        for tup in stanford_parse_tuples:
            if tup[0] == index_in_sentence:
                # Stop searching for a match
                break

        word = tup[1]

        # We explicitly pass None as node children, as they will be added
        # in a later iteration.
        node = DependencyNode(word, index_in_sentence, dependency, None)    
        tree.add(node, parent_index_in_sentence)

        children = tup[2]

        for child in children:
            dependency = child[0]
            children_indices = child[1]

            # There might be more than one child stored under the
            # same relation, so iterate over them
            for child_index in children_indices:
                queue.put((child_index, dependency, index_in_sentence))

    return tree