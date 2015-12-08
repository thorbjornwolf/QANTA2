from model import QANTA
import os
import cPickle

def run(input_folder, dimensionality):


    # dictionary with all info
    question_info_path = os.path.join(input_folder, "question_info")
    question_info_path = question_info_path #+ "_" + set_choice

    # dictionary with all the words
    vocabulary_path = os.path.join(input_folder, "vocabulary")
    vocabulary_path = vocabulary_path #+ "_" + set_choice

    # dictionary with all the dependencies
    dependency_path = os.path.join(input_folder, "dependency_vocabulary")
    dependency_path = dependency_path #+ "_" + set_choice

    # list of all the tree
    tree_list_path = os.path.join(input_folder, "tree_list")
    tree_list_path = tree_list_path #+ "_" + set_choice

    with open(vocabulary_path, 'rb') as vocabulary_file:
        vocabulary = cPickle.load(vocabulary_file)

    with open(dependency_path, 'rb') as dependency_file:
        dependency = cPickle.load(dependency_file)

    # with open(answers_path, 'rb') as answerfile:
    #     answers = cPickle.load(answerfile)

    with open(tree_list_path, 'rb') as tree_list_file:
        tree_list = cPickle.load(tree_list_file)

    qanta = QANTA(dimensionality, vocabulary, dependency)

    qanta.train(tree_list, n_incorrect_answers=100, n_epochs=1,
                    debug=True)


if __name__ == '__main__':
    source_path = 'parsed_data/hist'
    dimensionality = 50

    print "Running main_QANTA for data in {}".format(source_path)

    run(source_path, dimensionality)
