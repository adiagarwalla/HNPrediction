
filenames = ["svm_100k_predictions.txt", "knn_100k_predictions.txt", "ldasvm_100k_predictions.txt", "cosine_prediction_100k_classes.txt"]
	
file_test_target = open("test_100k_classes.txt", "r")
test_y = [int(l) for l in file_test_target.readlines()]
file_test_target.close()
results = [[0, 0 ,0],[0, 0 ,0],[0, 0 ,0]]

for filename in filenames:
	print "---------------------"
	print filename[:-4]
	f = open(filename, "r")
	predictions = [int(l) for l in f.readlines()]
	count = 0
	for i, j in zip(test_y, predictions):
		count += 1
		if filename[:-4] == "cosine_prediction_100k_classes":
			if count == 750:
				break
		results[i][j] += 1
	print results
	for classification in range(0,3):
		tp = 0
		tn = 0
		fp = 0
		fn = 0
		print "classification ", classification
		for target in range(0, 3):
			for prediction in range(0,3):
				if classification == target and classification == prediction:
					tp += results[target][prediction]
				elif classification == target and classification != prediction:
					fn += results[target][prediction]
				elif classification != target and classification == prediction:
					fp += results[target][prediction]
				else:
					tn += results[target][prediction]

		precision = tp/(tp + fp + 0.0)
		recall = tp/(tp + fn + 0.0)
		# print "fp", fp
		# print "fn", fn
		# print "tn", tn
		# print "tp", tp
		print "precision", precision
		print "recall", recall
		print "f1", 2*precision*recall/(precision + recall)