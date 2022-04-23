from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

print(len(stopwords.words("english")))
sentence = "This shouldn't be a problem. I will look into it personally."
new_words = []

words = list(word_tokenize(sentence))

new_words = [w for w in words if w not in stopwords.words("english")]
print(new_words)