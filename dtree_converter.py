from collections import namedtuple
from config import get_config
import csv_handler
import dtree_util
import cPickle   

    def dtree_from_orginal_qanta_to_dependency_tree():

        """
        Takes a path of the dtree file
        Returns a list of DependencyTrees.
        """

        list_of_dependency_trees = []

        path="C:\Users\zaki\Desktop\QANTA2-master\hist_split"

        vocab, rel_list, ans_list, tree_dict = cPickle.load(open(path, 'rb'))

        train_trees = tree_dict['train']

        ans_list = array([vocab.index(ans) for ans in ans_list])

        #rel_list.remove('root')

        for tree in train_trees:
            tree.ans_list = ans_list[ans_list != tree.ans_ind]

        for item in training_trees:

            ind, rel = tree.get(0).kids[0]
            root = tree.get(ind)

            dn=DependencyNode(None,root.qid)
            nodes=item.get_nodes()

            for node in nodes:
                DependencyNode(node.word, index_in_sentence, dependency, children=None, parent=None)

        return list_of_dependency_trees
