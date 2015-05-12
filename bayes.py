import nltk, re, pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, isdir, join
import numpy as np
import re
import sys
import getopt
import codecs
import time
from sklearn import naive_bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

file_extension = "100k"
if file_extension == "100k": 
    num_train = 89961
    num_test = 10039
    neg_train = -89961
else:
    num_train = 8974
    num_test = 1026
    neg_train = -8974

# reading a bag of words file back into python.
def read_bagofwords_dat(myfile, numofposts):
    bagofwords = np.fromfile(myfile, dtype=np.uint8, count=-1, sep="")
    bagofwords = np.reshape(bagofwords,(numofposts,-1))
    return bagofwords

def main():
    if file_extension == "100k":
        file_train = "../IgnoreData/train_100k_bag_of_words.dat"
    else:
        file_train = "train_"+file_extension+"_bag_of_words.dat"
    file_test = "test_"+file_extension+"_bag_of_words.dat"
    train = read_bagofwords_dat(file_train, num_train)
    test = read_bagofwords_dat(file_test, num_test)
    print "Read Bag of words files"

    file_titles = "train_"+file_extension+"_samples.txt"
    train_titles = []
    with open(file_titles) as f:
        train_titles = [line.rstrip() for line in f]

    print "Read train titles file"

    file_test_titles = "test_"+file_extension+"_samples.txt"
    test_titles = []
    with open(file_test_titles) as f:
        test_titles = [line.rstrip() for line in f]

    print "Read test titles file"

    file_train_classes = "train_"+file_extension+"_classes.txt"
    train_classes = []
    with open(file_train_classes) as f:
        train_classes = [line.rstrip() for line in f]

    file_test_classes = "test_"+file_extension+"_classes.txt"
    test_classes = []
    with open(file_test_classes) as f:
        test_classes = [line.rstrip() for line in f]

    print "Read test classes file whose length is"
    print len(test_classes)

    # cutoff = 2.5
    class_prior = [.88, .09, .03]
    # nb = GaussianNB()
    nb = MultinomialNB(1.0, False, class_prior)
    model = nb.fit(train, train_classes)

    y_pred = model.predict(test)

    outfile= open("bayes_multinomial_"+file_extension+"_classes.txt", 'w')
    outfile.write("\n".join(y_pred))
    outfile.close()

    print("Number of mislabeled test points out of a total %d points : %d" 
          % (len(y_pred),(test_classes != y_pred).sum()))

    # file_vocab = "train_"+file_extension+"_vocab.txt"
    # train_vocab = []
    # with open(file_vocab) as f:
    #     train_vocab = f.readlines()

    # train_vocab = [x.strip('\n') for x in train_vocab]

    # for i in range(0, num_test):
    #     if y_pred[i] != test_classes[i]:
    #         print y_pred[i], 
    #         for j in range(0, len(test[i])):
    #             if test[i][j] > cutoff: print train_vocab[j],
    #         print "\n -----------------------"


main()
