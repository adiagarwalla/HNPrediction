import lda
import numpy as np

# LDA on bag of words of titles across all posts
def lda(bagOfWords, topics, iterations, vocab, trainTitles, testBagOfWords, testTitles):
	model = lda.LDA(n_topics=topics, n_iter=iterations)
	model.fit(bagOfWords)

	#Distribution over vocabulary for each topic
	topic_word = model.topic_word_

	#Shape of topic_word
	print topic_word.shape

	# Top 5 vocab words per topic
	n = 5
	for i, topic_dist in enumerate(topic_word):
		topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
		print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))

	#Distribution over topics for each post
	post_topic = model.doc_topic_

	#Shape of post_topic
	print post_topic.shape

	# Top topic for 10 posts
	for n in range(10):
		topic_most_pr = post_topic[n].argmax()
		print("doc: {} topic: {}\n{}...".format(n,
                                            topic_most_pr,
                                            trainTitles[n][:50]))

	#Run fit transform for held out test data
	test_post_topic = model.fit_transform(testBagOfWords)

	#Make sure shape of this is correct
	print test_post_topic.shape

	#Distribution over topics for 10 posts in the test dataset
	for n in range(10):
		test_topic_most_pr = test_post_topic[n].argmax()
		print("test doc: {} topic: {}\n{}...".format(n, test_topic_most_pr, testTitles[n][:50]))

	print "end"

	#Prediction results for all the topics
	# results = []

	# for i, value in enumerate(testRow):
	# 	sumC = 0
	# 	for k in range(15):
	# 		sumC += doc_topic[value][k] * topic_word[k][testCol[i]]
 #        results.append(sumC)
    
 #    print results


if __name__ == "__main__":
    main(sys.argv[1:])
