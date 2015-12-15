from model import QANTA
import os
import cPickle

def run(input_folder, dimensionality, epochs):
	"""Opens up the relevant files and calls the model. Saves the model every
	10 epochs for later evaluation."""

	set_choice = "train"

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

	with open(tree_list_path, 'rb') as tree_list_file:
		tree_list = cPickle.load(tree_list_file)

	qanta = QANTA(dimensionality=dimensionality, vocabulary=vocabulary,
	 dependency_dict=dependency, embeddings_file="text8.model")

	for k in range(epochs/10):
		epoch_path = str((k+1)*10)
		qanta.train(tree_list, n_incorrect_answers=100, n_epochs=10) 
		temp_path = os.path.join(input_folder, epoch_path)
		with open(temp_path, 'wb') as f:
			cPickle.dump(qanta, f)
		print "Saved model in {}".format(temp_path)



if __name__ == '__main__':
	source_path = 'wiki' #directory of the preprocessed data
	dimensionality = 75 #dimensionality of wordvectors
	epochs = 200 #number of epochs to train over

	print "Running main_QANTA for data in {}".format(source_path)

	run(source_path, dimensionality, epochs)
