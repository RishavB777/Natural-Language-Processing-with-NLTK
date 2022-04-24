import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier

import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

documents = [(list(movie_reviews.words(fileid)), category) 
                        for category in movie_reviews.categories()
                        for fileid in movie_reviews.fileids(category)]
docs = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        docs.append((list(movie_reviews.words(fileid)), category))

random.shuffle(documents)

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

training_set = featuresets[:1900]
test_set = featuresets[1900:]

classifier_f = open('naivebayes.pickle','rb')
classifier = pickle.load(classifier_f)
classifier_f.close()


print("Original Naive Bayes Accuracy:", (nltk.classify.accuracy(classifier,test_set))*100)
classifier.show_most_informative_features(15)

MultinomialNB_Classifier = SklearnClassifier(MultinomialNB())
MultinomialNB_Classifier.train(training_set)
print("Multinomial Naive Bayes Accuracy:", (nltk.classify.accuracy(MultinomialNB_Classifier,test_set))*100)

"""
GaussianNB_Classifier = SklearnClassifier(GaussianNB())
GaussianNB_Classifier.train(training_set)
print("Gaussian Naive Bayes Accuracy:", (nltk.classify.accuracy(GaussianNB_Classifier,test_set))*100)
"""
BernoulliNB_Classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_Classifier.train(training_set)
print("Bernoulli Naive Bayes Accuracy:", (nltk.classify.accuracy(BernoulliNB_Classifier,test_set))*100)

LogisticRegression_Classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_Classifier.train(training_set)
print("Logistic Regression Accuracy:", (nltk.classify.accuracy(LogisticRegression_Classifier,test_set))*100)

# Stochastic Gradient Descent
SGD_Classifier = SklearnClassifier(SGDClassifier())
SGD_Classifier.train(training_set)
print("SGD Classifier Accuracy:", (nltk.classify.accuracy(SGD_Classifier,test_set))*100)

SVC_Classifier = SklearnClassifier(SVC())
SVC_Classifier.train(training_set)
print("SVC Classifier Accuracy:", (nltk.classify.accuracy(SVC_Classifier,test_set))*100)

LinearSVC_Classifier = SklearnClassifier(LinearSVC())
LinearSVC_Classifier.train(training_set)
print("LinearSVC Classifier Accuracy:", (nltk.classify.accuracy(LinearSVC_Classifier,test_set))*100)

NuSVC_Classifier = SklearnClassifier(NuSVC())
NuSVC_Classifier.train(training_set)
print("NuSVC Classifier Accuracy:", (nltk.classify.accuracy(NuSVC_Classifier,test_set))*100)


