import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import pickle
import numpy as np


n = int(input("no of sentences: "))
sentences = [str(input()) for _ in range(n)]
sentences = [entry.lower() for entry in sentences]

sentences = [word_tokenize(entry) for entry in sentences]
tagMap = defaultdict(lambda : wn.NOUN)
tagMap['J'] = wn.ADJ
tagMap['V'] = wn.VERB
tagMap['R'] = wn.ADV

for index,entry in enumerate(sentences):
    finalWords = []
    wordLemmatizer = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            wordFinal = wordLemmatizer.lemmatize(word,tagMap[tag[0]])
            finalWords.append(wordFinal)
    sentences[index] = str(finalWords)


clf = pickle.load(open("savedModel/nb/basic/model.sav", 'rb'))
Tfidf = pickle.load(open("savedModel/nb/basic/Tfidf.sav", 'rb'))
sentences = Tfidf.transform(sentences)
pred = clf.predict(sentences)
print(pred)