
import nltk, re, pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy
import re
import sys
import getopt
import codecs
import time
import json


chars = ['{','}','#','%','&','\(','\)','\[','\]','<','>',',', '!', '.', ';', 
'?', '*', '\\', '\/', '~', '_','|','=','+','^',':','\"','\'','@','-']

def stem(word):
    regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
    stem, suffix = re.findall(regexp, word)[0]
    return stem

def unique(a):
    """ return the list with duplicate elements removed """
    return list(set(a))

def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))

def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))



# reading a bag of words file back into python. The number and order
# of emails should be the same as in the *samples_class* file.
def read_bagofwords_dat(myfile, numofemails=10000):
    bagofwords = numpy.fromfile(myfile, dtype=numpy.uint8, count=-1, sep="")
    bagofwords=numpy.reshape(bagofwords,(numofemails,-1))
    return bagofwords

def get_class(points):
    thresholds = [500, 200, 100, 50, 20, 10]
    i = len(thresholds) 
    for threshold in thresholds:
      if points >= threshold:
        return str(i)
      i = i - 1
    return str(0)

def tokenize_corpus(inputf, train=True):
    porter = nltk.PorterStemmer() # also lancaster stemmer
    wnl = nltk.WordNetLemmatizer()
    stopWords = stopwords.words("english")
    classes = []
    samples = []
    docs = []
    if train == True:
        words = {}

    with open(inputf, "r") as f:
        for line in f:
            story = json.loads(line) #separate story on each line
            classes.append(get_class(story["points"])) # classification
            samples.append(story["objectID"]) 

            raw = story["title"]
            # remove noisy characters; tokenize
            raw = re.sub('[%s]' % ''.join(chars), ' ', raw)
            tokens = word_tokenize(raw)
            # convert to lower case
            tokens = [w.lower() for w in tokens]
            tokens = [w for w in tokens if w not in stopWords]
            tokens = [wnl.lemmatize(t) for t in tokens]
            tokens = [porter.stem(t) for t in tokens]
            if train == True:
                for t in tokens: 
                    # this is a hack but much faster than lookup each
                    # word within many dict keys
                    try:
                        words[t] = words[t]+1
                    except:
                        words[t] = 1
            docs.append(tokens)
    if train == True:
        return(docs, classes, samples, words)
    else:
        return(docs, classes, samples)
        

def wordcount_filter(words, num=5):
    keepset = []
    for k in words.keys():
        if(words[k] > num):
            keepset.append(k)
    print len(keepset)
    return(sorted(set(keepset)))


def find_wordcounts(docs, vocab):
    bagofwords = numpy.zeros(shape=(len(docs),len(vocab)), dtype=numpy.uint8)
    vocabIndex={}
    for i in range(len(vocab)):
       vocabIndex[vocab[i]]=i

    for i in range(len(docs)):
        doc = docs[i]

        for t in doc:
           index_t=vocabIndex.get(t)
           if index_t>=0:
              bagofwords[i,index_t]=bagofwords[i,index_t]+1

    print "Finished find_wordcounts for : "+str(len(docs))+"  docs"
    return(bagofwords)



def main(argv):
    inputf = ''
    vocabf = ''
    start_time = time.time()
    word_count_threshold = 30

    try:
        opts, args = getopt.getopt(argv,"i:v:",["ifile=","vocabfile="])
    except getopt.GetoptError:
        print 'python bagofwords.py -i <ifile> -v <vocabulary>'
        sys.exit(2)


    for opt, arg in opts:
        if opt == '-h':
            print 'bagofwords.py -i <ifile> -v <vocabulary>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputf = arg
        elif opt in ("-v", "--vocabfile"):
            vocabf = arg
	 
    outputf = inputf[:-4]

    if (not vocabf):
        (docs, classes, samples, words) = tokenize_corpus(inputf, train=True)
        vocab = wordcount_filter(words, num=word_count_threshold)
    else:
        vocabfile = open(vocabf, 'r')
        vocab = [line.rstrip('\n') for line in vocabfile]
        vocabfile.close()
        (docs, classes, samples) = tokenize_corpus(inputf, train=False)

    bow = find_wordcounts(docs, vocab)
    #sum over docs to see any zero word counts, since that would stink.
    x = numpy.sum(bow, axis=1) 
    print "doc with smallest number of words in vocab has: "+str(min(x))
    # print out files

    if (not vocabf):
        outfile= codecs.open(outputf+"_vocab.txt", 'w',"utf-8-sig")
        outfile.write("\n".join(vocab))
        outfile.close()
    #write to binary file for large data set
    bow.tofile(outputf+"_bag_of_words.dat")

    #write to text file for small data set
    #bow.tofile(outputf+"_bag_of_words.txt", sep=",", format="%s")
    outfile= open(outputf+"_classes.txt", 'w')
    outfile.write("\n".join(classes))
    outfile.close()
    outfile= open(outputf+"_samples.txt", 'w')
    outfile.write("\n".join(samples))
    outfile.close()
    print str(time.time() - start_time)

if __name__ == "__main__":
    main(sys.argv[1:])


