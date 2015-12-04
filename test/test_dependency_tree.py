# Run with `python -m unittest test.test_dependency_tree`

import unittest

from dependency_tree import tree_from_stanford_parse_tuples

class TestDependencyTree(unittest.TestCase):

    def test_get_ordered_words(self):

        sentence = [
            (0, None, [(u'root', [5])]),
            (1, u'For', []),
            (2, u'ten', []),
            (3, u'points', [(u'case', [1]), (u'nummod', [2])]),
            (5, u'identify', [(u'advcl', [17]), (u'nmod', [3]), (u'dobj', [8])]),
            (6, u'this', []),
            (7, u'English', []),
            (8, u'king', [(u'det', [6]), (u'nmod', [10]), (u'amod', [7])]),
            (9, u'of', []),
            (10, u'Wessex', [(u'case', [9]), (u'appos', [15])]),
            (12, u'the', []),
            (13, u'only', []),
            (14, u'English', []),
            (15, u'king', [(u'advmod', [13]), (u'det', [12]), (u'amod', [14])]),
            (16, u'to', []),
            (17, u'earn', [(u'xcomp', [22]), (u'mark', [16])]),
            (18, u'the', []),
            (19, u'epithet', [(u'det', [18])]),
            (21, u'the', []),
            (22, u'Great', [(u'dep', [19]), (u'det', [21])])
        ]

        vocabulary = {'earn':0,'English':1,'epithet':2,'For':3,'Great':4,
                 'identify':5,'king':6,'of':7,'only':8,'points':9,
                 'ten':10,'the':11,'this':12,'to':13,'Wessex':14}
        dep_dict = {'advcl':0,'advmod':1,'amod':2,'appos':3,'case':4,
                    'case':5,'dep':6,'det':7,'dobj':8,'mark':9,
                    'nmod':10,'nummod':11,'xcomp':12}
        qid = 500

        tree = tree_from_stanford_parse_tuples(sentence, qid, vocabulary, dep_dict)


        result = ' '.join(tree.get_ordered_words(vocabulary))
        expected = ('For ten points identify this English king of Wessex '
                    'the only English king to earn the epithet the Great')
        self.assertEquals(result, expected)

