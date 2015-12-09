#!/usr/bin/env python

"""This module evaluates an existing model on a selection of trees.
"""

import cPickle

from model import QANTA
from dependency_tree import DependencyTree, DependencyNode

def evaluate_model(qanta_model, trees):
    """Prints model accuracy, and returns it.

    qanta_model is an instance of model.QANTA
    trees is a list of DependencyTree
    """
    pred = qanta_model.predict_many(trees)
    fact = [t.answer for t in trees]

    n_correct = sum([1 for p,f in zip(pred, fact) if p==f])

    acc = float(n_correct)/len(fact)

    print "Model accuracy on {} trees: {:.4f}".format(len(trees), acc)
    return acc

if __name__ == '__main__':
    import argparse

    # command line arguments
    raw_args = argparse.ArgumentParser(
        description=('QANTA evaluation: Takes a model and some data, '
                     'and returns a score.'))
    raw_args.add_argument('model',
                          help="path to pickle of trained QANTA model",
                          type=str)
    raw_args.add_argument('data',
                          help=('path to pickled DependencyTrees, to be '
                                'used as evaluation data'), type=str)
    
    args = raw_args.parse_args()

    with open(args.model, 'rb') as f:
        model = cPickle.load(f)

    with open(args.data, 'rb') as f:
        trees = cPickle.load(f)

    evaluate_model(model, trees)