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


short_pos = open("short_reviews/positive.txt", "r").read()
short_neg = open("short_reviews/negative.txt", "r").read()

all_words = []

documents = []

# j is adjective, r is adverb, v is verb
# allowed_word_types = ["J","R",'V]
allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append((p,"pos"))
    words = nltk.tokenize.word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for p in short_neg.split('\n'):
    documents.append((p,"neg"))
    words = nltk.tokenize.word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

"""save_docs = open("pickled_algos/documents.pickle","wb")
pickle.dump(documents,save_docs)
save_docs.close()"""

save_docs = open("pickled_algos/all_words.pickle","wb")
pickle.dump(all_words,save_docs)
save_docs.close()

pickle_1 = open('pickled_algos/documents.pickle','rb')
documents = pickle.load(pickle_1)
pickle_1.close()

short_pos_words = nltk.tokenize.word_tokenize(short_pos)
short_neg_words = nltk.tokenize.word_tokenize(short_neg)


all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

save_word_features = open("pickled_algos/word_features.pickle","wb")
pickle.dump(word_features,save_word_features)
save_word_features.close()


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

pickle_1 = open('pickled_algos/MultinomialNB_Classifier.pickle','rb')
MultinomialNB_Classifier = pickle.load(pickle_1)
pickle_1.close()

"""MultinomialNB_Classifier = SklearnClassifier(MultinomialNB())
MultinomialNB_Classifier.train(training_set)"""

print("Multinomial Naive Bayes Accuracy:", (nltk.classify.accuracy(MultinomialNB_Classifier,test_set))*100)



"""save_classifier = open("pickled_algos/MultinomialNB_Classifier.pickle","wb")
pickle.dump(MultinomialNB_Classifier,save_classifier)
save_classifier.close()"""

pickle_1 = open('pickled_algos/BernoulliNB_Classifier.pickle','rb')
BernoulliNB_Classifier = pickle.load(pickle_1)
pickle_1.close()


"""BernoulliNB_Classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_Classifier.train(training_set)"""

print("Bernoulli Naive Bayes Accuracy:", (nltk.classify.accuracy(BernoulliNB_Classifier,test_set))*100)

"""save_classifier = open("pickled_algos/BernoulliNB_Classifier.pickle","wb")
pickle.dump(BernoulliNB_Classifier,save_classifier)
save_classifier.close()"""

pickle_1 = open('pickled_algos/LogisticRegression_Classifier.pickle','rb')
LogisticRegression_Classifier = pickle.load(pickle_1)
pickle_1.close()

#LogisticRegression_Classifier = SklearnClassifier(LogisticRegression())
#LogisticRegression_Classifier.train(training_set)
print("Logistic Regression Accuracy:", (nltk.classify.accuracy(LogisticRegression_Classifier,test_set))*100)

"""save_classifier = open("pickled_algos/LogisticRegression_Classifier.pickle","wb")
pickle.dump(LogisticRegression_Classifier,save_classifier)
save_classifier.close()"""

# Stochastic Gradient Descent

pickle_1 = open('pickled_algos/SGD_Classifier.pickle','rb')
SGD_Classifier = pickle.load(pickle_1)
pickle_1.close()

#SGD_Classifier = SklearnClassifier(SGDClassifier())
#GD_Classifier.train(training_set)
print("SGD Classifier Accuracy:", (nltk.classify.accuracy(SGD_Classifier,test_set))*100)

"""save_classifier = open("pickled_algos/SGD_Classifier.pickle","wb")
pickle.dump(SGD_Classifier,save_classifier)
save_classifier.close()"""

pickle_1 = open('pickled_algos/SVC_Classifier.pickle','rb')
SVC_Classifier = pickle.load(pickle_1)
pickle_1.close()


"""SVC_Classifier = SklearnClassifier(SVC())
SVC_Classifier.train(training_set)"""
print("SVC Classifier Accuracy:", (nltk.classify.accuracy(SVC_Classifier,test_set))*100)

"""save_classifier = open("pickled_algos/SVC_Classifier.pickle","wb")
pickle.dump(SVC_Classifier,save_classifier)
save_classifier.close()"""

pickle_1 = open('pickled_algos/LinearSVC_Classifier.pickle','rb')
LinearSVC_Classifier = pickle.load(pickle_1)
pickle_1.close()

"""LinearSVC_Classifier = SklearnClassifier(LinearSVC())
LinearSVC_Classifier.train(training_set)"""
print("LinearSVC Classifier Accuracy:", (nltk.classify.accuracy(LinearSVC_Classifier,test_set))*100)

"""save_classifier = open("pickled_algos/LinearSVC_Classifier.pickle","wb")
pickle.dump(LinearSVC_Classifier,save_classifier)
save_classifier.close()"""

pickle_1 = open('pickled_algos/NuSVC_Classifier.pickle','rb')
NuSVC_Classifier = pickle.load(pickle_1)
pickle_1.close()

"""NuSVC_Classifier = SklearnClassifier(NuSVC())
NuSVC_Classifier.train(training_set)"""
print("NuSVC Classifier Accuracy:", (nltk.classify.accuracy(NuSVC_Classifier,test_set))*100)

"""save_classifier = open("pickled_algos/NuSVC_Classifier.pickle","wb")
pickle.dump(NuSVC_Classifier,save_classifier)
save_classifier.close()"""


voted_classifier = VoteClassifier(classifier,MultinomialNB_Classifier,BernoulliNB_Classifier,SGD_Classifier,
                                    SVC_Classifier,LinearSVC_Classifier,NuSVC_Classifier)


print("Voted Classifier accuracy percent:",(nltk.classify.accuracy(voted_classifier,test_set))*100)

def sentiment(text):
    feats = find_features(text)

    return voted_classifier.classify(feats), voted_classifier.confidence(feats)