import json


def get_score(points):
    points = int(points)
    return int(points/20)

def get_hr(created_at):




random.seed(0) # use same seed everytime
train_x = []
train_y = []
test_x = []
test_y = []
file = "" #edit
#change from file to stream!!!!
f = open(filename, "r")
json = json.load(f)
for story in json["hits"]:
    train =  true
    # set aside 10% of data for test
    if (random.random() < .1) train = false
    features = []    
    features.append(get_hr(story["created_at"]))
    features.append(get_author_id(story["author"])
    features.append(get_url(story["url"]))
    if train:
        train_x.append(features)
        train_y.append(get_score(story["points"]))
    else:
        test_x.append(features)
        test_y.append(get_score(story["points"]))

f.close()

print(train_x)
print(train_y)
print(test_x)
