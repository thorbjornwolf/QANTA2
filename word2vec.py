import csv
import cPickle
from gensim.models import word2vec

def append_sentence(skip_head=1, sub_delimiter=' ||| '):
    """Creates one long sentence out of all the testanswer questions
    and appends it to the original trainset for word2vec"""

    filen = "./output/stanford_parsed"

    question_information_path = "./output/question_info"

    sentence_id_path = "./output/sentence_ID"

    with open(filen, 'rb') as f:
        input = cPickle.load(f)

    with open(question_information_path, 'rb') as f:
        question_info = cPickle.load(f)

    with open(sentence_id_path, 'rb') as f:
        answers = cPickle.load(f)

    sentence = ""

    for k in range(len(input)):
        sentence_length = len(input[k])/2
        for l in range(sentence_length):
            sentence += str(" ") + str(input[k][l+1][1]).lower()

        #print k
        sentence += str(question_info[answers[k]][2]).lower() + str(" ")

        for l in range(sentence_length-1):
            #print input[k][l+sentence_length]
            sentence += str(" ") + str(input[k][l+1+sentence_length][1]).lower() 

    #print sentence

    with open ("data/text8", "a") as f:
        f.write(sentence)

def train():
    sentences = word2vec.Text8Corpus('data/text8')
    model = word2vec.Word2Vec(sentences, size=75)

    model.save('data/text8.model')

    model.save_word2vec_format('data/text8.model.bin', binary=True)

append_sentence()
train()