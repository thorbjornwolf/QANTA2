# Run by going to /QANTA2/ and running 'python -m unittest test.test_stanford_parse'

import unittest

from stanford_parse import dependency_parse

class TestParseResult(unittest.TestCase):

    def test_result(self):
        expected_results = [
            [(0, None, [(u'root', [14])]), (1, u'After', []), (2, u'you', []), 
             (3, u"'ve", []), (4, u'installed', [(u'aux', [3]), (u'nsubj', [2]), 
                (u'dobj', [7]), (u'mark', [1])]), (5, u'the', []), 
             (6, u'Stanford', []), (7, u'Parser', [(u'det', [5]), (u'nmod', [11]),
                (u'compound', [6])]), (8, u'in', []), (9, u'your', []), 
             (10, u'home', []), (11, u'directory', [(u'case', [8]), 
                (u'nmod:poss', [9]), (u'compound', [10])]), (13, u'just', []), 
             (14, u'use', [(u'advmod', [13]), (u'dobj', [17]), 
                (u'advcl', [4, 19])]), (15, u'this', []), (16, u'python', []), 
             (17, u'recipe', [(u'det', [15]), (u'compound', [16])]), 
             (18, u'to', []), (19, u'get', [(u'dobj', [23]), (u'mark', [18])]), 
             (20, u'the', []), (21, u'flat', []), (22, u'bracketed', 
                [(u'amod', [21])]), (23, u'parse', [(u'det', [20]), (u'amod', [22])])],

            [(0, None, [(u'root', [5])]), (1, u'Could', []), (2, u'your', []), 
             (3, u'computerphobic', []), (4, u'cousin', [(u'nmod:poss', [2]), 
                (u'amod', [3])]), (5, u'tell', [(u'aux', [1]), (u'ccomp', [8]), 
                (u'nsubj', [4])]), (6, u'what', []), (7, u'is', []), (8, u'going', 
                [(u'aux', [7]), (u'compound:prt', [9]), (u'nmod', [12]), 
                (u'nsubj', [6])]), (9, u'on', []), (10, u'in', []), 
                (11, u'the', []), (12, u'code', [(u'case', [10]), 
                    (u'det', [11])])],

            [(0, None, [(u'root', [3])]), (1, u'this', []), (2, u'is', []), 
             (3, u'that', [(u'cc', [4]), (u'cop', [2]), (u'conj', [7]), 
                (u'nsubj', [1])]), (4, u'and', []), (5, u'that', []), 
             (6, u'is', []), (7, u'this', [(u'cop', [6]), (u'nsubj', [5])])]
        ]

        inputs = ["After you've installed the Stanford Parser in your home directory, just use this python recipe to get the flat bracketed parse",
              "Could your computerphobic cousin tell what is going on in the code?",
              "this is that and that is this"]

        self.assertEqual(expected_results, dependency_parse(inputs))

if __name__ == '__main__':
    unittest.main()