from model import QANTA
import os
import cPickle

def Something(input_folder):

	dimension = 50 #change dimension!



    # dictionary with all info
    question_info_path = os.path.join(input_folder, "question_info")
    question_info_path = question_info_path + "_" + set_choice

    # dictionary with all the words
    vocabulary_path = os.path.join(input_folder, "vocabulary")
    vocabulary_path = vocabulary_path + "_" + set_choice

    # dictionary with all the dependencies
    dependency_path = os.path.join(input_folder, "dependency_vocabulary")
    dependency_path = dependency_path + "_" + set_choice

    # list of all the tree
    tree_list_path = os.path.join(input_folder, "tree_list")
    tree_list_path = tree_list_path + "_" + set_choice

	with open(vocabulary_path, 'rb') as vocabulary_file:
		vocabulary = cPickle.load(vocabulary_file)

	with open(dependency_path, 'rb') as dependency_file:
		dependency = cPickle.load(dependency_file)

	with open(answers_path, 'rb') as answerfile:
		answers = cPickle.load(answerfile)

	with open(tree_list_path, 'rb') as tree_list_file:
		tree_list = cPickle.load(tree_list_file)

	print tree_list[23]

	qanta = QANTA(dimension, vocabulary, dependency)

	qanta.train(tree_list, 100, 1)

Something("./output")