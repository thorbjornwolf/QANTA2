# Import unused. Remove it.
from collections import defaultdict
import csv_handler

# See if you can find a different name than 'main.py'
# It sounds so preposterous, especially when it is
# only supposed to handle preprocessing.

#Not working yet, probably many errors

class Config:
	csv_file = None

	@staticmethod
	def Init(config):
		if 'csv_file' in config:
			Config.csv_file = config['csv_file']

def preprocessing(path):

	questions=[]
	questions_list=[]
	
 	# Parse the questions from the CSV file, outputs a list
 	questions = csv_handler.parse_question_csv(Config.csv_file)
	
 	# For better manageability
 	for item in questions:
 		# questions_list is simply a list, it has no method `add_question`
 		# You could, as mentioned below, use a namedtuple to store question lines,
 		# and then `append` those to this list
 		questions_list.add_question(item[0], item[1], item[2], item[3] , item[4])
	
 	# Parse the data using Stanford parser and then add it to the dependency tree
	load_status = 0
 	for text, question_id in questions_list:

 		load_status += 1
 		parsed_tree = stanford_parse.dependency_parse(text)

 		if load_status % 1000 == 0:
			print load_status 

		# This class simply wraps the DependencyTree constructor (__init__)
		# Might as well remove it.
		Create_tree.Init(parsed_tree, question_id, answer_index)
 	
 	#########################################
 	          TO BE CONTINUED
 	#########################################


# This class doesn't seem to be needed
# Just remove it
class Create_tree(object):
	Dependency_tree = None

	def Init(tree):
		Dependency_tree = Dependency_tree(tree, question_id, answer_index)

# Python class naming style is DependencyTree
class dependency_tree(object):

	def __init__(self, tree, question_id, answer_index):

		# Wait initializing the node until you have something ready to 
		# put in it. This gives an error.
		self.root = Dependency_tree_node()
		self.answer_index = answer_index
		self.question_id = question_id
		# You already know the number of nodes: len(tree)
		# Initialize this directly, instead of to 0
		self.n_nodes = 0

		# I am not sure about the order of the nodes. Better err on the safe
		# side and construct the tree by a queue: If the node has a place in
		# the tree, i.e. is another nodes child, insert it. Otherwise put it
		# back in the queue. This is a bit inefficient, but it will get the 
		# job done. If you can think of a better way, please go ahead!
		for node in tree:
			# You already know the number of nodes: len(tree)
			# No need to += 1 for each iteration
			self.n_nodes+=1
			print node
			# DependencyNode.Add_child is not implemented yet.
			# Also, code style dictates that we use DependencyNode.add_child
			# instead.
			self.root.Add_child(surface, uris)

# Python class naming style is DependencyNode (also, see dev notes)
class dependency_tree_node(object):
	
	def __init__(self, word_index, dependency_index):

		self.word_index = word_index
		self.dependency_index = dependency_index
		# children should be a list, not a set or dict.
		self.children = {}

# Use a namedtuple for this, instead of a full class. They can be found in 
# collections
class Data:

	# You forgot a colon at the end here :)
	def __init__(self, question_id, fold, answer, category, text)

		self.question_id = question_id
		self.fold = fold
		self.answer = answer
		self.category = category
		self.text = text

# Overall suggestion:
# Use divide and conquer: Develop small section, and write some tests
# to see if they work as expected. If you run a file as `python main.py`, any
# method calls out in the open, like
main()
# will be called properly. Use this to do some quick testing, e.g. to make
# sure that the DependencyTree works like it should
