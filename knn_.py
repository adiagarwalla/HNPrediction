from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import numpy as np

file_extension = "10k"

if file_extension == "100k": 
	num_train = 89961
	num_test = 10039
else:
	num_train = 8974
	num_test = 1026

num_clusters = 15
num_neighbors = 10

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


	# read training into cluster
	model_knn = KNeighborsClassifier(n_neighbors = num_neighbors)
	model_knn.fit(train_x, train_y)
	predictions = model_knn.predict(test_x)
	score = sum([1 if i == j else 0 for i, j in zip(predictions, test_y)])/(num_test + 0.0)
	print "KNN score: ", + score
	out = open("knn_"+file_extension+"_predictions.txt", "w")
	for prediction in predictions:
		out.write(str(prediction) + "\n")
	out.close()


main()
