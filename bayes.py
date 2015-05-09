import nltk, re, pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, isdir, join
import numpy
import re
import sys
import getopt
import codecs
import time
from sklearn import naive_bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB


num_train = 45000
num_test = 5000
# reading a bag of words file back into python.
def read_bagofwords_dat(myfile, numofposts):
    bagofwords = np.fromfile(myfile, dtype=np.uint8, count=-1, sep="")
    bagofwords = np.reshape(bagofwords,(numofposts,-1))
    return bagofwords

def main():
    # Bag of words for training data
    file_train = "../train_emails_bag_of_words_200.dat"

    # Bag of words for test data
    file_test = "../test_emails_bag_of_words_0.dat"
    train = read_bagofwords_dat(file_train, num_train)
    test = read_bagofwords_dat(file_test, num_test)

    train_target = []
    for i in range(0, num_train):
        if i < num_train/2:
            train_target.append("notspam")
        else:
            train_target.append("spam")

    test_target = []
    for i in range(0, num_test):
        if i < num_test/2:
            test_target.append("notspam")
        else:
            test_target.append("spam")

    cutoff = 2.5
    class_prior = [.2, .8]
    #nb = GaussianNB()
    #nb = MultinomialNB(1.0, False, class_prior)
    nb = BernoulliNB(1.0, cutoff, False, class_prior)
    model = nb.fit(train, train_target)

    y_pred = model.predict(test)

    print("Number of mislabeled test points out of a total %d points : %d" 
          % (len(y_pred),(test_target != y_pred).sum()))

    vocab = []
    with open("trec07p_data/Test/train_emails_vocab_200.txt") as f:
        vocab = f.readlines()

    vocab = [x.strip('\n') for x in vocab]

    for i in range(0, num_test):
        if y_pred[i] != test_target[i]:
            print y_pred[i], 
            for j in range(0, len(test[i])):
                if test[i][j] > cutoff: print vocab[j],
            print "\n -----------------------"


main()