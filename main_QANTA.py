from model import QANTA
import os
import cPickle

def Something(input_folder):

	dimension = 50 #change dimension!

	train_folder = os.path.join(process_dir, "train")

	stanford_parsed_path = os.path.join(train_folder, "stanford_parsed") 
	vocabulary_path = os.path.join(train_folder, "vocabulary")
	dependency_path = os.path.join(train_folder, "dependency_vocabulary")
	tree_list_path = os.path.join(train_folder, "tree_list")

	test_folder = os.path.join(process_dir, "test")

	stanford_parsed_path = os.path.join(test_folder, "stanford_parsed") 
	vocabulary_path = os.path.join(test_folder, "vocabulary")
	dependency_path = os.path.join(test_folder, "dependency_vocabulary")
	tree_list_path = os.path.join(test_folder, "tree_list")

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