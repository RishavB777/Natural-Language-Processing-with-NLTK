import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier

import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

documents = [(list(movie_reviews.words(fileid)), category) 
                        for category in movie_reviews.categories()
                        for fileid in movie_reviews.fileids(category)]


short_pos = open("short_reviews/positive.txt", "r").read()
short_neg = open("short_reviews/negative.txt", "r").read()

documents = []
for r in short_pos.split('\n'):
    documents.append((r,"pos"))

for r in short_neg.split('\n'):
    documents.append((r,"neg"))

all_words = []

short_pos_words = nltk.tokenize.word_tokenize(short_pos)
short_neg_words = nltk.tokenize.word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())


all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

def find_features(doc):
    features = {}
    words = nltk.tokenize.word_tokenize(doc)
    for w in word_features:
        features[w] = (w in words)
    return features


featuresets = [(find_features(rev),category) for (rev,category) in documents]


training_set = featuresets[:10000]
test_set = featuresets[10000:]


classifier_f = open('naivebayes.pickle','rb')
classifier = pickle.load(classifier_f)
classifier_f.close()


print("Original Naive Bayes Accuracy:", (nltk.classify.accuracy(classifier,test_set))*100)

MultinomialNB_Classifier = SklearnClassifier(MultinomialNB())
MultinomialNB_Classifier.train(training_set)
print("Multinomial Naive Bayes Accuracy:", (nltk.classify.accuracy(MultinomialNB_Classifier,test_set))*100)

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


voted_classifier = VoteClassifier(classifier,MultinomialNB_Classifier,BernoulliNB_Classifier,SGD_Classifier
                                    ,LinearSVC_Classifier,NuSVC_Classifier)


print("Voted Classifier accuracy percent:",(nltk.classify.accuracy(voted_classifier,test_set))*100)

