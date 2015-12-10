
import dtree_util
import cPickle   

def dtree_from_orginal_qanta_to_dependency_tree(original_data, target_path):

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


if __name__ == '__main__':
    import argparse

    # command line arguments
    raw_args = argparse.ArgumentParser(
        description=('QANTA converter from original format to QANTA 2 format'))
    raw_args.add_argument('source_path',
                          help="path to pickle of original QANTA trees and params",
                          type=str)
    raw_args.add_argument('target_path',
                          help=('path to directory in which we store the '
                                'converted data'), type=str)
    
    args = raw_args.parse_args()

    with open(args.source_path, 'rb') as f:
        source = cPickle.load(f)

    dtree_from_orginal_qanta_to_dependency_tree(source, args.target_path)