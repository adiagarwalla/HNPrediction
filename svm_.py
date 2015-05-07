from sklearn import svm
import lda
import numpy as np

file_extension = "100k"

if file_extension == "100k": 
	num_train = 89961
	num_test = 10039
else:
	num_train = 8974
	num_test = 1026

num_topics = 15
num_iterations = 200

def read_bagofwords_dat(myfile, numofposts):
    bagofwords = np.fromfile(myfile, dtype=np.uint8, count=-1, sep="")
    bagofwords = np.reshape(bagofwords,(numofposts,-1))
    return bagofwords

def main ():
	filename_train = "train_"+file_extension+"_bag_of_words.dat"
	filename_test = "test_"+file_extension+"_bag_of_words.dat"
	file_train_target = open("train_"+file_extension+"_classes.txt", "r")
	file_test_target = open("test_"+file_extension+"_classes.txt", "r")
	train_x = read_bagofwords_dat(filename_train, num_train)
	test_x = read_bagofwords_dat(filename_test, num_test)
	train_y = [int(l) for l in file_train_target.readlines()]
	test_y = [int(l) for l in file_test_target.readlines()]

	model_svm = svm.LinearSVC()
	model_svm.fit(train_x, train_y)
	print "SVM on bag of words:", model_svm.score(test_x, test_y)

	# run again on LDA transformed data
	model_lda = lda.LDA(n_topics=num_topics, n_iter=num_iterations)
	model_lda.fit_transform(train_x)
	model_lda.transform(train_y)
	model_svm = svm.LinearSVC()
	model_svm.fit(train_x, train_y)
	print "SVM on LDA transformed:", model_svm.score(test_x, test_y)

main()
