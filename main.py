from collections import defaultdict

#Not working yet, probably many errors

def main(path):

	questions=[]
	questions_list=[]

 	# Parse the questions from the CSV file, outputs a list
 	questions = csv_handler.parse_question_csv("C:\Users\zaki\Desktop\QANTA2-master\questions20k.csv")

 	# For better manageability
 	for item in questions:
 		questions_list.add_question(item[0], item[1], item[2], item[3] , item[4])
	
 	# Parse the data using Stanford parser and then add it to the dependency tree
	load_status = 0
 	for text, question_id in questions_list:

 		load_status += 1
 		parsed_tree = stanford_parse.dependency_parse(text)

 		if load_status % 1000 == 0:
			print load_status 

		Create_tree.Init(parsed_tree, question_id, answer_index)
 	
 	#########################################
 	          TO BE CONTINUED
 	#########################################



class Create_tree(object):
	Dependency_tree = None

	def Init(tree):
		Dependency_tree = Dependency_tree(tree, question_id, answer_index)

class dependency_tree(object):

	def __init__(self, tree, question_id, answer_index):

		self.root = Dependency_tree_node()
		self.answer_index = answer_index
		self.question_id = question_id
		self.n_nodes = 0

		for node in tree:
			self.n_nodes+=1
			print node
			self.root.Add_child(surface, uris)

class dependency_tree_node(object):
	
	def __init__(self, word_index, dependency_index):

		self.word_index = word_index
		self.dependency_index = dependency_index
		self.children = {}

class Data:

	def __init__(self, question_id, fold, answer, category, text)

		self.question_id = question_id
		self.fold = fold
		self.answer = answer
		self.category = category
		self.text = text
