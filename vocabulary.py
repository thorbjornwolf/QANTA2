import pickle

def vocabulary(filen):
	"""Takes a file as input, unpickles it and add every entity 
	of it to the vocabulary. Returns vocabulary."""

	with open (filen, 'rb') as f:
		input =  pickle.load(f)

	vocab = {}
	dep_vocab = {}

	for k in range(len(input)):
		for l in range(len(input[k])):
			for m in [0,2]:
				if input[k][l][m][0] in vocab:
					pass
				else:
					print "added " + input[k][l][m][0]
					vocab[input[k][l][m][0]] = len(vocab) + 1
				if input[k][l][1][0] in dep_vocab:
					dep_vocab[[k][l][1][0]] = len(dep_vocab) + 1

	return vocab