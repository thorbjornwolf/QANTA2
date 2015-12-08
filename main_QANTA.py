from model import QANTA
import os
import cPickle

def run(input_folder, dimensionality):

	train_folder = os.path.join(input_folder, "train")

	stanford_parsed_path = os.path.join(train_folder, "stanford_parsed") 
	vocabulary_path = os.path.join(train_folder, "vocabulary")
	dependency_path = os.path.join(train_folder, "dependency_vocabulary")
	tree_list_path = os.path.join(train_folder, "tree_list")

	test_folder = os.path.join(input_folder, "test")

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

	qanta = QANTA(dimensionality, vocabulary, dependency)

	qanta.train(tree_list, n_incorrect_answers=100, n_epochs=1,
					print_training_accuracy=True)


if __name__ == '__main__':
	source_path = 'output-hist'
	dimensionality = 50

	print "Running main_QANTA for data in {}".format(source_path)

	run(source_path, dimensionality)
