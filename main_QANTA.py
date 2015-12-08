from model import QANTA
import os
import cPickle

def run(input_folder, dimensionality):

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

	qanta = QANTA(dimensionality, vocabulary, dependency)

	qanta.train(tree_list, n_incorrect_answers=100, n_epochs=1,
					print_training_accuracy=True)


if __name__ == '__main__':
	source_path = 'output-hist'
	dimensionality = 50

	print "Running main_QANTA for data in {}".format(source_path)

	run(source_path, dimensionality)