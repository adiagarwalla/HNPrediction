import ijson
from ijson import items
import codecs

# for easy lookup of story ids
stories = dict()
num = "100k"
f_stories = open("train_" + num + "_ids.txt", "r")
for story_id in f_stories:
    stories[int(story_id)] = 0
f_stories.close()

f_stories = open("test_" + num + "_ids.txt", "r")
for story_id in f_stories:
    stories[int(story_id)] = 0
f_stories.close()

print "finished reading in stories"

comments = dict()
#f = open("../HNCommentsTest.json", "r")
f = open("../HNCommentsAll.json", 'r')
objects= items(f, 'item.hits.item')

print "finished creating json objects"

for comment in objects:
    text = comment["comment_text"]
    story_id = comment["story_id"]
    # only care about comments that belong to one of our stories
    if not stories.has_key(story_id):
        continue
    if comments.has_key(story_id): 
        comments[story_id] += text
    else:
        comments[story_id] = text


print "finished creating dictionary"

f_comments = codecs.open("comments.txt", 'w',"utf-8-sig")

for (key, value) in comments.items():
    f_comments.write(str(key) + " " + value + "\n")
    
f_comments.close()
f.close()