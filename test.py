import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower()    )

def bag_of_words(tokenized_words, all_words):
    pass



# Testing the code functionality
# stri = "How long will shipping take?"
# print(stri)
# print(tokenize(stri))
# token = tokenize(stri)
# l = []
# for x in token:
#     l.append(stem(x))
    
# print(l)