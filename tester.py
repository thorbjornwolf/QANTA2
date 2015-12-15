import cPickle
import os
from model import QANTA


input_folder = "history"
set_choice = "test"

tree_list_path = os.path.join(input_folder, "tree_list")
tree_list_path = tree_list_path + "_" + set_choice

model_path = "history/700"

with open(tree_list_path, 'rb') as tree_list_file:
    tree_list = cPickle.load(tree_list_file)

with open(model_path, 'rb') as model_file:
    model = cPickle.load(model_file)

print model

#print tree_list

print model.get_accuracy(tree_list, 1)
print model.get_accuracy(tree_list, 3)
print model.get_accuracy(tree_list, 5)