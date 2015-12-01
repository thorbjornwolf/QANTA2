# Run by going to /QANTA2/ and running 'python -m unittest test.test_csv_handler'

import csv
import os
import tempfile
import unittest

from csv_handler import parse_question_csv

tmpfile = 'test/tmp.csv'

class TestParseResult(unittest.TestCase):

    def test_result(self):
        header = ['col0', 'col1', 'col2', 'col3', 'col4']
        expected_results = [ 
                     ['a', 'b', 'c', 'd', ['r','h','g','j']],
        ]
        inp = [x for x in expected_results[0]]
        inp[4] = ' ||| '.join(inp[4])

        with open(tmpfile, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(inp)

        self.assertEqual(expected_results, parse_question_csv(tmpfile))

        os.remove(tmpfile)

    def test_errors(self):
        inps = ["col0,col1,col2,col3,col4\na,b,c,d,e,f", 
               "col0,col1,col2,col3,col4\na,b",
               'col0,col1,col2,col3,col4\na,b,c,d,"e ||| f"',
               'col0,col1,col2,col3,col4\na,b,c,d,"e ||| f ||| q ||| w ||| e ||| r ||| t ||| y ||| u ||| i ||| o"',]

        for inp in inps:
            with open(tmpfile, 'w') as f:
                f.write(inp)
            
            self.assertRaises(AssertionError, parse_question_csv, tmpfile)

        os.remove(tmpfile)
        

if __name__ == '__main__':
    unittest.main()