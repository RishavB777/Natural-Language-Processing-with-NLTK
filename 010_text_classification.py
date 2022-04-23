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

print(len(all_words))

# Frequency Distribution
all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15))
print(len(all_words))