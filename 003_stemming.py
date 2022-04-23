from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()
sent = "python Pythoneer Pythonista Pythonly Pythoning"

words = word_tokenize(sent)
stemmed_words = [stemmer.stem(w) for w in words]
print(stemmed_words)