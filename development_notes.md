

# QANTA from scratch

## Preprocessing
- [x] Parse CSV (using built-in csv module)
- [x] Ensure division in train, test sets is kept
- [x] Apply Stanford Parser to question sentences, get dependency tree as text
    + [ ] Check whether it drops sentences when it is out of memory
    + [ ] Check whether it splits sentence strings that are, in fact, two or more sentences. This happens in the question in line 136 in the 20k dataset.
- [x] Build a word vocabulary that maps words to indices (later to be used in fetching word embeddings).
    + Either as a list where vocabulary.index(word) gives an index, or as a dict, where vocabulary[word] gives an index.
    + Make sure to include the answer strings in the vocab. They need to have their own embedded representations.
    + Consider whether answers should have spaces replaced with underscores, or whether we just keep them as a string with spaces.
- [x] Build a dependency list (in fact an ordered set or vocabulary; only unique entries) that allows mapping a dependency string to an integer (later to be used for fetching the dependency matrix)
    + Same remarks as for the word vocabulary
- [ ] Convert Stanford Parser dependency tree text to actual tree data structure
    + [ ] Create and use a class `DependencyNode`:
        * has `DependencyNode.word_index` (the word's index in the vocabulary) 
        * has `DependencyNode.dependency_index` (the index in the dependency list of the dependency between this node and its parent)
        * has `DependencyNode.children` (list of nodes)
    + [ ] Create and use a class `DependencyTree`:
        * has `DependencyTree.root` (the top node)
        * has `DependencyTree.answer_index` (the answer string's index in the vocabulary)
        * has `DependencyTree.question_id`, a uniquely identifying question id (expected to be given by the input CSV)
        * has `DependencyTree.n_nodes()`, giving the number of nodes in the tree
- [ ] Figure out a meaningful way of storing trees, word vocabulary, dependency list

## Model training
- [x] Create a QANTAModel class with
    + A vector (word embedding) for each word in the vocabulary
    + A matrix (dependency embedding) for each dependency in the dependency list
    + The global ``additional matrix'' W_v (eq. 1 description)
    + The global bias term
- [x] Initialize word and dependency embeddings, and the additional matrix and bias
- [ ] Look into what is considered best practice for initialization of vectors and matrices
- [ ] Consider initializing as many words as possible with precomputed vectors, such as those from word2vec
- [ ] Remove stopwords c.f. the original qanta code.
- [x] Implement a method (e.g. `QANTAModel.calculate_embedding`) that can calculate a DependencyTree's embedded representation (following eq. 4)
- [ ] Figure out why the original QANTA implementation does not use the `rank` calculation (eq 5)
- [ ] Build an overall method `QANTAModel.train` that, given a list of `DependencyTree`s (and possibly a list of wrong answers?) will train the model.
    + [x] Implemented a method `QANTA.sentence_error` that calculates eq. 5, given
        * a sentence tree
        * a list of incorrect answers
    + [x] Calculate objective function (eq. 6) (model.py:94)
    + [ ] Do backpropagation through structure / AdaGrad (eq. 7 and supplementary reading)
- [ ] Build an overall method `QANTAModel.predict` that, given a tree returns the most likely answer (from the set of possible answers)
- [ ] Actually train a model on the training set
	
## Evaluation
- [ ] Evaluate the trained model's performance on the test set.

## General checklist
- Did you document your code well? Could your girlfriend/parent/computerphobic cousin tell what is going on in the code?
- Did you write unit tests for the parts you developed? Did you push them?
- Did you update this checklist?
