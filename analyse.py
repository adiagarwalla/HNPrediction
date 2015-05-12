filenames = [#"svm_100k_predictions.txt", "knn_100k_predictions.txt", 
#"ldasvm_100k_predictions.txt", "cosine_prediction_100k_classes.txt", "dt_100k_predictions.txt"
"bayes_gaussian_100k_classes.txt", "bayes_multinomial_100k_classes.txt"]

	
file_test_target = open("test_100k_classes.txt", "r")
test_y = [int(l) for l in file_test_target.readlines()]
file_test_target.close()

for filename in filenames:
	results = [[0, 0 ,0],[0, 0 ,0],[0, 0 ,0]]
	print "---------------------"
	print filename[:-4]
	f_prediction = open(filename, "r")
	predictions = [int(l) for l in f_prediction.readlines()]
	f_prediction.close()
	count = 0
	for i, j in zip(test_y, predictions):
		count += 1
		#if filename[:-4] == "cosine_prediction_100k_classes":
			#if count == 750:
		#		break
		results[i][j] += 1
	print results
	for classification in range(0,3):
		tp = 0.0
		tn = 0.0
		fp = 0.0
		fn = 0.0
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

		assert(tp + tn + fp + fn == len(predictions))

		if tp + fp == 0:
			precision = float("NaN")
		else:
			precision = tp/(tp + fp)
		recall = tp/(tp + fn)
		# print "fp", fp
		# print "fn", fn
		# print "tn", tn
		# print "tp", tp
		print "precision", precision
		print "recall", recall
		print "f1", 2*precision*recall/(precision + recall)
