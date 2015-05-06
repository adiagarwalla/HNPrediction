import lda
import numpy as np

file_extension = "100k"
if file_extension == "100k": 
	num_train = 89961
	num_test = 10039
else:
	num_train = 8974
	num_test = 1026


number_topics = 15
number_iterations = 200

# reading a bag of words file back into python.
def read_bagofwords_dat(myfile, numofposts):
    bagofwords = np.fromfile(myfile, dtype=np.uint8, count=-1, sep="")
    bagofwords = np.reshape(bagofwords,(numofposts,-1))
    return bagofwords

# LDA on bag of words of titles across all posts
def ldaPost(bagOfWords, topics, iterations, vocab, trainTitles, testBagOfWords, testTitles):
	model = lda.LDA(n_topics=topics, n_iter=iterations)
	model.fit(bagOfWords)

	print "Fit train bag of words and the shape of the topic-word distribution is - "

	#Distribution over vocabulary for each topic
	topic_word = model.topic_word_

	#Shape of topic_word
	print topic_word.shape

	print "Top 5 vocab words per topic"
	# Top 5 vocab words per topic
	n = 5
	for i, topic_dist in enumerate(topic_word):
		topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
		print('*Topic {}\n {}'.format(i, ' '.join(topic_words)))

	#Distribution over topics for each post
	post_topic = model.doc_topic_

	print "Shape of trainDocument-topic distribution is - "
	#Shape of post_topic
	print post_topic.shape

	print "Top topic for 10 posts are - "
	# Top topic for 10 posts
	for n in range(10):
		topic_most_pr = post_topic[n].argmax()
		print("doc: {} topic: {}\n{}...".format(n,
                                            topic_most_pr,
                                            trainTitles[n][:50]))

	#Run fit transform for held out test data
	test_post_topic = model.fit_transform(testBagOfWords)

	outfile= open("test_doc_topic.txt", 'w')
        outfile.write(test_post_topic)
        outfile.close()

	print "Fit transformed test bag of words and shape of document - topic distribution is - "
	#Make sure shape of this is correct
	print test_post_topic.shape

	print "Top topic for 10 posts are - "
	#Distribution over topics for 10 posts in the test dataset
	for n in range(10):
		test_topic_most_pr = test_post_topic[n].argmax()
		print("test doc: {} topic: {}\n{}...".format(n, test_topic_most_pr, testTitles[n][:50]))

	print "end"

def main():
	file_train = "train_"+file_extension+"_bag_of_words.dat"
	file_test = "test_"+file_extension+"_bag_of_words.dat"
	train = read_bagofwords_dat(file_train, num_train)
	test = read_bagofwords_dat(file_test, num_test)

	print "Read Bag of words files"

	file_vocab = "train_"+file_extension+"_vocab.txt"
	train_vocab = []
        with open(file_vocab) as f:
            train_vocab = f.readlines()

        print "Read train vocabulary file"

        file_titles = "train_"+file_extension+"_samples.txt"
        train_titles = []
        with open(file_titles) as f:
            train_titles = f.readlines()

        print "Read train titles file"

        file_test_titles = "test_"+file_extension+"_samples.txt"
        test_titles = []
        with open(file_test_titles) as f:
            test_titles = f.readlines()

        print "Read test titles file"


        best = -float('Inf')
        best_num_topics = 5
        for topics in range(5, 30, 5):
        	print "--------------------------------------------------"
        	print "Number of topics is = " + str(topics)
        	model = lda.LDA(n_topics=topics, n_iter=number_iterations)
        	model.fit(train)
        	print model.loglikelihood()
        	if model.loglikelihood() > best:
        		best_num_topics = topics
        		best = model.loglikelihood()
        
        print "best number of topics =" + str(best_num_topics)
        number_topics = best_num_topics	

        ldaPost(train, number_topics, number_iterations, train_vocab, train_titles, test, test_titles)

main()
