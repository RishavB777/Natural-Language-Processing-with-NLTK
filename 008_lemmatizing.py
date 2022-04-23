from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
lst = ["cats","his","was","pythonly","better","geese"]
"""for i in lst:
    print(lemmatizer.lemmatize(i))

"""
# pos(part of speech) is by default set as 'n' (noun)
print(lemmatizer.lemmatize("better",pos="a")) # pos = adjective
print(lemmatizer.lemmatize("better",pos="v")) # pos = verb
print(lemmatizer.lemmatize("better",pos="n")) # pos = noun