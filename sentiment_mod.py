import nltk

import pickle

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


# j is adjective, r is adverb, v is verb
# allowed_word_types = ["J","R",'V]
allowed_word_types = ["J"]

pickle_1 = open('pickled_algos/all_words.pickle','rb')
all_words = pickle.load(pickle_1)
pickle_1.close()

pickle_1 = open('pickled_algos/documents.pickle','rb')
documents = pickle.load(pickle_1)
pickle_1.close()

short_pos_words = nltk.tokenize.word_tokenize(short_pos)
short_neg_words = nltk.tokenize.word_tokenize(short_neg)


all_words = nltk.FreqDist(all_words)

pickle_1 = open('pickled_algos/word_features.pickle','rb')
word_features = pickle.load(pickle_1)
pickle_1.close()


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


print("Multinomial Naive Bayes Accuracy:", (nltk.classify.accuracy(MultinomialNB_Classifier,test_set))*100)


pickle_1 = open('pickled_algos/BernoulliNB_Classifier.pickle','rb')
BernoulliNB_Classifier = pickle.load(pickle_1)
pickle_1.close()


print("Bernoulli Naive Bayes Accuracy:", (nltk.classify.accuracy(BernoulliNB_Classifier,test_set))*100)


pickle_1 = open('pickled_algos/LogisticRegression_Classifier.pickle','rb')
LogisticRegression_Classifier = pickle.load(pickle_1)
pickle_1.close()


print("Logistic Regression Accuracy:", (nltk.classify.accuracy(LogisticRegression_Classifier,test_set))*100)


pickle_1 = open('pickled_algos/SGD_Classifier.pickle','rb')
SGD_Classifier = pickle.load(pickle_1)
pickle_1.close()

print("SGD Classifier Accuracy:", (nltk.classify.accuracy(SGD_Classifier,test_set))*100)


pickle_1 = open('pickled_algos/SVC_Classifier.pickle','rb')
SVC_Classifier = pickle.load(pickle_1)
pickle_1.close()


print("SVC Classifier Accuracy:", (nltk.classify.accuracy(SVC_Classifier,test_set))*100)

pickle_1 = open('pickled_algos/LinearSVC_Classifier.pickle','rb')
LinearSVC_Classifier = pickle.load(pickle_1)
pickle_1.close()


print("LinearSVC Classifier Accuracy:", (nltk.classify.accuracy(LinearSVC_Classifier,test_set))*100)


pickle_1 = open('pickled_algos/NuSVC_Classifier.pickle','rb')
NuSVC_Classifier = pickle.load(pickle_1)
pickle_1.close()

print("NuSVC Classifier Accuracy:", (nltk.classify.accuracy(NuSVC_Classifier,test_set))*100)


voted_classifier = VoteClassifier(classifier,MultinomialNB_Classifier,BernoulliNB_Classifier,SGD_Classifier,
                                    SVC_Classifier,LinearSVC_Classifier,NuSVC_Classifier)


print("Voted Classifier accuracy percent:",(nltk.classify.accuracy(voted_classifier,test_set))*100)

def sentiment(text):
    feats = find_features(text)

    return voted_classifier.classify(feats), voted_classifier.confidence(feats)