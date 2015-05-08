import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import heapq
import math

file_extension = "10k"
if file_extension == "100k": 
	num_train = 89961
	num_test = 10039
	neg_train = -89961
else:
	num_train = 8974
	num_test = 1026
	neg_train = -8974

def read_bagofwords_dat(myfile, numofposts):
    bagofwords = np.fromfile(myfile, dtype=np.uint8, count=-1, sep="")
    bagofwords = np.reshape(bagofwords,(numofposts,-1))
    return bagofwords

def cosSim(trainBagOfWords, testBagOfWords, trainTitles, testTitles, trainClasses, testClases):
	#Tf-idf conversion
	tfidf_transformer = TfidfTransformer()

	print "Fit and transform the trainBagOfWords"
	tfidf_matrix_train = tfidf_transformer.fit_transform(trainBagOfWords).toarray()
	print tfidf_matrix_train.shape

	print "Fit and transform testBagOfWords"
	tfidf_matrix_test = tfidf_transformer.transform(testBagOfWords).toarray()
	print tfidf_matrix_test.shape

	count = 0
	results = []
	# Iterate through every row in test file
	for testVector in tfidf_matrix_test:
		x = cosine_similarity(testVector, tfidf_matrix_train)
		max_indices = np.argsort(x[0])[-25:][::-1]
		max_values = heapq.nlargest(25, x[0])
		# if (count < 10):
		print testTitles[count]
		max_values_norm = [float(i) / sum(max_values) for i in max_values]
		max_values_norm_nonan = []

		for normalized_value in max_values_norm:
			if math.isnan(normalized_value):
				max_values_norm_nonan.append(0)
			else:
				max_values_norm_nonan.append(normalized_value)

		sumPredict = 0
		for ind, cos_k in enumerate(max_indices):
			if trainClasses[cos_k] == "0":
				factor = 0
			elif trainClasses[cos_k] == "1":
				factor = 10
			else:
				factor = 100

			sumPredict += max_values_norm_nonan[ind] * factor
		
			# print "Top 25 similar indices for this post " + str(max_indices)
			# print "Top 25 similar posts for this post "
			# for similarIndex in max_indices:
			# 	print similarIndex
			# 	print trainTitles[similarIndex]
		count += 1
		results.append(sumPredict)

	final_results = []
	for result in results:
		if result < 10:
			final_results.append("0")
		elif result < 100:
			final_results.append("1")
		else:
			final_results.append("2")

	print "Final results done"
	print len(final_results)

	accuracy_count = 0
	for i, value in enumerate(final_results):
		if value == testClases[i]:
			accuracy_count +=1

	print final_results
	print accuracy_count
	print float(accuracy_count)/num_test

	outfile= open("cosine_prediction_"+file_extension+"_classes.txt", 'w')
	outfile.write("\n".join(final_results))
	outfile.close()

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
	
	# print "Converted numpy bag of words to sparse matrices"
	# train_sparse = csr_matrix(train)
	# test_sparse = csr_matrix(test)

	cosSim(train, test, train_titles, test_titles, train_classes, test_classes)

main()
