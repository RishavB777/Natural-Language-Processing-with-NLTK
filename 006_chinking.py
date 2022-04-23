from nltk.tokenize import word_tokenize,PunktSentenceTokenizer
from nltk.corpus import state_union
import nltk

train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')

tokenizer = PunktSentenceTokenizer(train_text)
sents = tokenizer.tokenize(sample_text)

for sent in sents:
    words = word_tokenize(sent)
    tagged = nltk.pos_tag(words)
    chunkGram = r"""Chunk: {<.*>+}
                        }<VB.?|TO|IN>+{""" # Chinking means the chunks without }<THESE>{
    chunkParser = nltk.RegexpParser(chunkGram)
    chunked = chunkParser.parse(tagged)
    chunked.draw()   
  