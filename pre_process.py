import csv
import cPickle
import os
import nltk
import re
from datetime import datetime
from nltk.parse import stanford
from config import get_config
from dependency_tree import tree_from_stanford_parse_tuples



def parse_question_csv(csv_path, target_path=None, skip_head=1, sub_delimiter=' ||| '):
	"""Builds a list of lists, each of which represents a line in the csv.
	Note that the 5th element in each line is itself a list.

	csv_path is the path to the .csv file on your system
	skip_head is the number of lines to skip in the beginning of the file
	sub_delimiter is the extra delimiter in the 5th column of the file

	If target_path is defined, cPickles the result to that file.
	Otherwise returns the result.
	"""

	csv_questions = []
	with open(csv_path) as f:
		handle = csv.reader(f, strict=True)
		for _ in xrange(skip_head):
			handle.next()

		for i, line in enumerate(handle):
			assert len(line) == 5
			line[4] = line[4].split(sub_delimiter)  # Question text
			assert 0 < len(line[4]) < 12, "Error in line {}".format(i + 2)
			csv_questions.append(line)

	with open(target_path, 'wb') as f:
		cPickle.dump(csv_questions, f)


def questions_to_sentences(csv_pickle, set_choice, sentence_ID_path,
						   sentences_path, answers_path, question_info_path):
	"""Takes the raw CSV-text as input and outputs each sentence from all the 
	questions. Another list containing the question ID is also outputed."""

	with open(csv_pickle, 'rb') as csvfile:
		csv_questions = cPickle.load(csvfile)

	# Quick and dirty
	sent_replacements = (('1/2', ''),)

	temp_answers = []
	sentence_ID = []
	sentences = []
	answers = []
	question_information = {}

	for questions in csv_questions:
		if questions[1] == set_choice and 'History' in questions[2]:
			#temp = questions[3].split()
			temp_string = questions[3]
			questions[3] = questions[3].replace(" ", "_")
			question_information[questions[0]] = [questions[1], questions[2], questions[3]]
			
			if questions[3] not in answers:
				answers.append(questions[3])
				temp_answers.append(temp_string)

	for questions in csv_questions:
		if questions[1] == set_choice and 'History' in questions[2]:
			for sentence in questions[4]:
				for k in range(len(temp_answers)):
					sentence = sentence.replace(temp_answers[k], answers[k])
				for string, sub in sent_replacements:
					sentence = sentence.replace(string, sub)
				sentences.append(sentence)
				sentence_ID.append(questions[0])

				#sentences.append(sentence)
				#sentence_ID.append(questions[0])

	with open(sentence_ID_path, 'wb') as f:
		cPickle.dump(sentence_ID, f)

	with open(sentences_path, 'wb') as f:
		cPickle.dump(sentences, f)

	with open(answers_path, 'wb') as f:
		cPickle.dump(answers, f)

	with open(question_info_path, 'wb') as f:
		cPickle.dump(question_information, f)


def node_converter(node, target_path=None):
	"""Convert node to triple:
			address, word, deps
			where deps is a list
					where each element is a tuple
							(dependency_name, [address_dep1, address_dep2, ...])
	"""
	address = node[1]['address']
	word = node[1]['word']
	deps = []
	for k, v in node[1]['deps'].iteritems():
		deps.append((k, v))
	return (address, word, deps)


def dependency_parse(sentences_path, target_path=None):
	"""sentences is a list of strings

	Returns a list, where each element corresponds to the sentence
	in the same index in the input list. The elements are themselves
	lists of tuples: (index_in_sentence, word, dependencies),
	where dependencies are the tuple (dependency_name, [dep1, dep2, ...])
			dep1, dep2 being index_in_sentence for the dependent words
	"""

	# Initialize target file so we can append to it later
	with open(target_path, 'wb') as f:
		cPickle.dump("", f)

	with open(sentences_path, 'rb') as sentencesfile:
		sentences1 = cPickle.load(sentencesfile)

	config = get_config('Stanford Parser')
	# E.g. /usr/local/Cellar/stanford-parser/3.5.2/libexec/stanford-parser.jar
	os.environ['STANFORD_PARSER'] = config['STANFORD_PARSER']
	# E.g.
	# /usr/local/Cellar/stanford-parser/3.5.2/libexec/stanford-parser-3.5.2-models.jar
	os.environ['STANFORD_MODELS'] = config['STANFORD_MODELS']

	parser = stanford.StanfordDependencyParser()
	# We can set java options through java_options. They default to '-mx1000m'

	for k in range(len(sentences1)/100):
		print "Batch " + str(k + 1) + " of " + str((len(sentences1)+2)/100)
	
		sentences = sentences1[k*100:(k+1)*100]		

		parsed = parser.raw_parse_sents(sentences)

		output = []
		for sentence in parsed:
			depgraph = list(sentence)
			assert len(depgraph) == 1
			depgraph = depgraph[0]

			root_address = depgraph.root['address']
			nodes = map(node_converter, depgraph.nodes.items())

			output.append(nodes)

		with open(target_path, 'ab') as f:
			cPickle.dump(output, f)


def vocabulary(filen, answers_path, vocabulary_path=None, dependency_path=None):
	"""Takes a file as input, unpickles it and add every entity 
	of it to the vocabulary. Pickles vocabulary."""

	with open(filen, 'rb') as f:
		input = cPickle.load(f)

	with open(answers_path, 'rb') as f:
		answers = cPickle.load(f)

	vocab = {}
	dep_vocab = {}

	for k in range(len(input)):
		for l in range(len(input[k])):

			if input[k][l][1] in vocab:
				pass
			else:
				vocab[input[k][l][1]] = len(vocab)

			for m in range(len(input[k][l][2])):
				if input[k][l][2][m][0] in dep_vocab:
					pass

				else:
					dep_vocab[input[k][l][2][m][0]] = len(dep_vocab)

	for element in answers:
		if element not in vocab:
			vocab[element] = len(vocab)

	with open(vocabulary_path, 'wb') as f:
		cPickle.dump(vocab, f)

	with open(dependency_path, 'wb') as f:
		cPickle.dump(dep_vocab, f)


def create_tree(sentences_path, sentence_ID_path, question_info_path,
				vocabulary_path, dependency_path, stanford_parsed_path, tree_list_path):
	"""Opens up all the data and passes it to dependency_tree to create all trees"""

	tree_list = []

	with open(sentences_path, 'rb') as f:
		sentences = cPickle.load(f)

	with open(sentence_ID_path, 'rb') as f:
		sentences_ID = cPickle.load(f)

	with open(question_info_path, 'rb') as f:
		question_info = cPickle.load(f)

	with open(vocabulary_path, 'rb') as f:
		vocabulary = cPickle.load(f)

	with open(dependency_path, 'rb') as f:
		dependency = cPickle.load(f)

	with open(stanford_parsed_path, 'rb') as f:
		stanford_parsed = cPickle.load(f)

	for k in range(len(sentences)):
		answer = question_info[sentences_ID[k]][2]
		tree = tree_from_stanford_parse_tuples(stanford_parsed[k], answer,
											   vocabulary, dependency)
		
		tree_list.append(tree)

	with open(tree_list_path, 'wb') as f:
		cPickle.dump(tree_list, f)


def process(csv_file, output_file, set_choice, process_dir, start_from):
	"""Starts from whatever index that is specified in start_from. 
	Basically takes the csv file and outputs all the data structure needed
	for the model"""
	# CSV imported
	parsed_csv_path = os.path.join(process_dir, "parsed_csv")
	# all the sentence_IDs in a list
	sentence_ID_path = os.path.join(process_dir, "sentence_ID")
	# all sentences in a list
	sentences_path = os.path.join(process_dir, "sentences")
	# answers in a list
	answers_path = os.path.join(process_dir, "answers") 
	# dictionary with all info
	question_info_path = os.path.join(process_dir, "question_info")
	# All the stanford parsed sentences
	stanford_parsed_path = os.path.join(process_dir, "stanford_parsed")
	# dictionary with all the words
	vocabulary_path = os.path.join(process_dir, "vocabulary")
	# dictionary with all the dependencies
	dependency_path = os.path.join(process_dir, "dependency_vocabulary")
	# list of all the tree
	tree_list_path = os.path.join(process_dir, "tree_list")

	if start_from <= 1:
		parse_question_csv(csv_file, parsed_csv_path)
	if start_from <= 2:
		questions_to_sentences(parsed_csv_path, set_choice, sentence_ID_path,
							   sentences_path, answers_path, question_info_path)
	if start_from <= 3:
		dependency_parse(sentences_path, stanford_parsed_path)
	if start_from <= 4:
		vocabulary(stanford_parsed_path, answers_path,
				   vocabulary_path, dependency_path)
	if start_from <= 5:
		create_tree(sentences_path, sentence_ID_path, question_info_path,
					vocabulary_path, dependency_path, stanford_parsed_path, tree_list_path)


def main():
	import argparse

	# command line arguments
	raw_args = argparse.ArgumentParser(
		description='QANTA preprocessing: Going from CSV question files to QANTA format')
	raw_args.add_argument('-s', '--source', dest='source_file',
						  help='location of source CSV file',
						  type=str, default="./questions.csv")
	raw_args.add_argument('-o', '--output', dest='output_file',
						  help='location of output file', type=str)
	raw_args.add_argument('--set-choice', dest='set_choice',
						  help='what type of set (train, test, dev) to preprocess', type=str, default="train")
	raw_args.add_argument('-d', '--directory', dest='process_dir',
						  help='Location of directory in which we store stepwise results',
						  default=None, type=str)
	raw_args.add_argument('--start-point', dest='start_point', type=int, default=0,
						  help=('For starting over from somewhere within the '
								'main process. If == 0, the full process is '
								'run. Otherwise, the number indicates where '
								'in the the process method we should start. '
								'(See source code for full reference)'))

	args = raw_args.parse_args()

	if not args.process_dir:
		args.process_dir = 'qanta-preprocess-{}'.format(
			datetime.now().strftime('%Y-%m-%d--%H-%M-%S'))
		pass
	if not os.path.exists(args.process_dir):
		os.makedirs(args.process_dir)

	process(csv_file=args.source_file, output_file=args.output_file,
			set_choice=args.set_choice,
			process_dir=args.process_dir, start_from=args.start_point)

if __name__ == '__main__':
	main()
