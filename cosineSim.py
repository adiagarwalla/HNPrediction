import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import heapq

file_extension = "10k"
if file_extension == "100k": 
	num_train = 89961
	num_test = 10039
else:
	num_train = 8974
	num_test = 1026

def read_bagofwords_dat(myfile, numofposts):
    bagofwords = np.fromfile(myfile, dtype=np.uint8, count=-1, sep="")
    bagofwords = np.reshape(bagofwords,(numofposts,-1))
    return bagofwords

def cosSim(trainBagOfWords, testBagOfWords):
	#Tf-idf conversion
	tfidf_transformer = TfidfTransformer()

	print "Fit and transform the trainBagOfWords"
	tfidf_matrix_train = tfidf_transformer.fit_transform(trainBagOfWords).toarray()
	print tfidf_matrix_train.shape

	print "Fit and transform testBagOfWords"
	tfidf_matrix_test = tfidf_transformer.transform(testBagOfWords).toarray()
	print tfidf_matrix_test.shape

	count = 0
	# Iterate through every row in test file
	for testVector in tfidf_matrix_test:
		x = cosine_similarity(testVector, tfidf_matrix_train)
		max_indices = np.argsort(x[0])[-25:][::-1]
		max_values = heapq.nlargest(25, x[0])
		if (count < 10):
			print testVector
			print "Top 25 similar indices for this post " + str(max_indices)
			print "Top 25 similar values for this post " + str(max_values)
		count += 1

def main():
	file_train = "train_"+file_extension+"_bag_of_words.dat"
	file_test = "test_"+file_extension+"_bag_of_words.dat"
	train = read_bagofwords_dat(file_train, num_train)
	test = read_bagofwords_dat(file_test, num_test)

	print "Read Bag of words files"

	# print "Converted numpy bag of words to sparse matrices"
	# train_sparse = csr_matrix(train)
	# test_sparse = csr_matrix(test)

	cosSim(train, test)

main()
