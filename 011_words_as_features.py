import nltk
import random
from nltk.corpus import movie_reviews

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
features = {}
word_features = list(all_words.keys())[:3000]
def find_features(doc):
    words = set(doc)
    for w in word_features:
        features[w] = (w in words)
    return features

print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
