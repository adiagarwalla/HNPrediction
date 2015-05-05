import json
import random


random.seed(0)
limit = 100000
num = str(limit/1000)
f_train = open("train_" + num + "k.txt", 'w')
f_test = open("test_" + num + "k.txt", 'w')

f = open("../HNStoriesAll.json", 'r')


jsonobj = json.load(f)
num_test = 0
num_train = 0
print "finished loading json"
for hit in jsonobj:
    for story in hit["hits"]:
        if num_test + num_train == limit:
            break

        if random.random() < .1 : 
            train = False
        else:
            train = True


        if train:
            json.dump(story, f_train)
            f_train.write("\n")
            num_train += 1

        else:
            json.dump(story, f_test)
            f_test.write("\n")
            num_test += 1

print "Training data size:", num_train
print "Testing data size:", num_test

f_train.close()
f_test.close()
f.close()

# random.seed(0) # use same seed everytime
# train_x = []
# train_y = []
# test_x = []
# test_y = []
# file = "" #edit
# #change from file to stream!!!!
# f = open(filename, "r")
# json = json.load(f)
# for story in json["hits"]:
#     train =  true
#     # set aside 10% of data for test
#     if (random.random() < .1) train = false
#     #features = []    
#     #features.append(get_hr(story["created_at"]))
#     #features.append(get_author_id(story["author"]) 
#     #features.append(get_url(story["url"]))
#     bagofwords = get_bagofwords(story["title"])
#     if train:
#         #train_x.append(features)
#         train_x.append(bagofwords)
#         train_y.append(get_score(story["points"]))
#     else:
#         #test_x.append(features)
#         test_x.append(bagofwords)
#         test_y.append(get_score(story["points"]))

# f.close()

# print(train_x)
# print(train_y)
# print(test_x)
