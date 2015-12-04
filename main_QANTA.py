from model import *
import os
import cPickle

def Something(input_folder):

	sentence_ID_path = os.path.join(input_folder, "sentence_ID")
	sentences_path = os.path.join(input_folder, "sentences")
	answers_path = os.path.join(input_folder, "answers")
	question_info_path = os.path.join(input_folder, "question_info")
	stanford_parsed_path = os.path.join(input_folder, "stanford_parsed")
	vocabulary_path = os.path.join(input_folder, "vocabulary")
	dependency_path = os.path.join(input_folder, "dependency_path")

	with open(vocabulary_path, 'rb') as vocabulary_file:
		vocabulary = cPickle.load(vocabulary_file)

	with open(dependency_path, 'rb') as dependency_file:
		dependency = cPickle.load(dependency_file)

	with open(answers_path, 'rb') as answerfile:
		answers = cPickle.load(answerfile)

	qanta = QANTA(50, vocabulary, dependency, answers)

Something("./output")