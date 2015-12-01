import pickle

def vocabulary(filen)
	with open (filen, 'rb') as f:
		input =  pickle.load(f)

	vocab = {}

	for k in range(len(input)):
		for l in range(len(input[k])):
			for m in [0,2]:
				if input[k][l][m][0] in vocab:
					pass
				else:
					print "added " + input[k][l][m][0]
					vocab[input[k][l][m][0]] = len(vocab) + 1
