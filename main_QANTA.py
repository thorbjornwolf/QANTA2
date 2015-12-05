from model import QANTA
import os
import cPickle

def Something(input_folder):

	dimension = 50 #change dimension!

	sentence_ID_path = os.path.join(input_folder, "sentence_ID") #all the sentence_IDs in a list
	sentences_path = os.path.join(input_folder, "sentences") #all sentences in a list
	answers_path = os.path.join(input_folder, "answers") #answers in a list
	question_info_path = os.path.join(input_folder, "question_info") #dictionary with all info
	stanford_parsed_path = os.path.join(input_folder, "stanford_parsed") 
	vocabulary_path = os.path.join(input_folder, "vocabulary")
	dependency_path = os.path.join(input_folder, "dependency_vocabulary")
	tree_list_path = os.path.join(input_folder, "tree_list")

	with open(vocabulary_path, 'rb') as vocabulary_file:
		vocabulary = cPickle.load(vocabulary_file)

	with open(dependency_path, 'rb') as dependency_file:
		dependency = cPickle.load(dependency_file)

	with open(answers_path, 'rb') as answerfile:
		answers = cPickle.load(answerfile)

	with open(tree_list_path, 'rb') as tree_list_file:
		tree_list = cPickle.load(tree_list_file)

	print tree_list[23]

	qanta = QANTA(dimension, vocabulary, dependency, answers)

	qanta.train(tree_list)

Something("./output")