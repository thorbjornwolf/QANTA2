#!/usr/bin/env python

import os

from nltk.parse import stanford

from config import get_config


def node_converter(node):
    """Convert node to triple:
        address, word, deps
        where deps is a list
            where each element is a tuple
                (dependency_name, [address_dep1, address_dep2, ...])
    """
    address = node[1]['address']
    word = node[1]['word']
    deps = []
    for k,v in node[1]['deps'].iteritems():
        deps.append((k,v))
    return (address, word, deps)

def dependency_parse(sentences):
    """sentences is a list of strings

    Returns a list, where each element corresponds to the sentence
    in the same index in the input list. The elements are themselves
    lists of tuples: (index_in_sentence, word, dependencies),
    where dependencies are the tuple (dependency_name, [dep1, dep2, ...])
        dep1, dep2 being index_in_sentence for the dependent words
    """
    config = get_config('Stanford Parser')
    # E.g. /usr/local/Cellar/stanford-parser/3.5.2/libexec/stanford-parser.jar
    os.environ['STANFORD_PARSER'] = config['STANFORD_PARSER']
    # E.g. /usr/local/Cellar/stanford-parser/3.5.2/libexec/stanford-parser-3.5.2-models.jar
    os.environ['STANFORD_MODELS'] = config['STANFORD_MODELS']

    parser = stanford.StanfordDependencyParser(java_options='-mx10000m')
    # We can set java options through java_options. They default to '-mx1000m'
    # Here set to ~10GB

    parsed = parser.raw_parse_sents(sentences)

    output = []
    for sentence in parsed:
        depgraph = list(sentence)
        assert len(depgraph) == 1
        depgraph = depgraph[0]

        root_address = depgraph.root['address']
        nodes = map(node_converter, depgraph.nodes.items())
            
        output.append(nodes)

    return output