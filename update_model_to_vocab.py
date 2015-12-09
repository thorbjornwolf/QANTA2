#!/usr/bin/env python

from utils import Vocabulary
from model import QANTA
import cPickle
import numpy as np

def update_model(model):

    d = model.dimensionality
    if not type(model.vocabulary) == Vocabulary:
        print "Updating vocabulary and We"
        model.vocabulary = Vocabulary(model.vocabulary)
        # Append to We
        unkn_word_embedding = np.random.uniform(-1, 1, size=(1, d))
        model.We = np.append(model.We, unkn_word_embedding, axis=0)

    if not type(model.dependency_dict) == Vocabulary:
        print "Updating dependency_dict and Wr"
        model.dependency_dict = Vocabulary(model.dependency_dict)
        unkn_relation_embedding = np.random.uniform(-1, 1, size=(1, d, d))
        model.Wr = np.append(model.Wr, unkn_relation_embedding, axis=0)

    return model

if __name__ == '__main__':
    import argparse

    # command line arguments
    raw_args = argparse.ArgumentParser(
        description=('QANTA Model vocabulary style updater '
                     'makes your model use Vocabulary'))
    raw_args.add_argument('model',
                          help="path to pickle of trained QANTA model",
                          type=str)
    
    args = raw_args.parse_args()


    with open(args.model, 'rb') as f:
        model = cPickle.load(f)

    oldmodel = args.model + '_deprecated'
    print "Making backup: Old model is saved as {}".format(oldmodel)
    import shutil
    shutil.copyfile(args.model, oldmodel)

    model = update_model(model)

    with open(args.model, 'wb') as f:
        cPickle.dump(model, f)