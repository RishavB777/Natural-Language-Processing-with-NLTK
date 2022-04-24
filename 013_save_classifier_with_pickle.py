import nltk
import random
from nltk.corpus import movie_reviews
import pickle

documents = [(list(movie_reviews.words(fileid)), category) 
                        for category in movie_reviews.categories()
                        for fileid in movie_reviews.fileids(category)]
docs = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        docs.append((list(movie_reviews.words(fileid)), category))

random.shuffle(documents)

#print(documents[1]) positive

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(doc):
    features = {}
    words = set(doc)
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets = [(find_features(rev),category) for (rev,category) in documents]
#print(featuresets)
"""for (rev,category) in documents[:1]:
    print(f"{rev},{category}")
"""

training_set = featuresets[:1900]
test_set = featuresets[1900:]

#classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open('naivebayes.pickle','rb')
classifier = pickle.load(classifier_f)
classifier_f.close()


print("Naive Bayes Accuracy:", (nltk.classify.accuracy(classifier,test_set))*100)
classifier.show_most_informative_features(15)

"""
save_classifier = open("naivebayes.pickle","wb") #mode = write in bytes
pickle.dump(classifier, save_classifier)
save_classifier.close()"""