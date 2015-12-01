import csv

def parse_question_csv(csv_path, skip_head=1, sub_delimiter=' ||| '):
    """Builds a list of lists, each of which represents a line in the csv.
    Note that the 5th element in each line is itself a list.

    csv_path is the path to the .csv file on your system
    skip_head is the number of lines to skip in the beginning of the file
    sub_delimiter is the extra delimiter in the 5th column of the file

    If target_path is defined, cPickles the result to that file.
    Otherwise returns the result.
    """

    result = []
    with open(csv_path) as f:
        handle = csv.reader(f, strict=True)
        for _ in xrange(skip_head):
            handle.next()

        for i, line in enumerate(handle):
            assert len(line) == 5
            line[4] = line[4].split(sub_delimiter) # Question text
            assert 0 < len(line[4]) < 12, "Error in line {}".format(i + 2)
            result.append(line)

    return result
