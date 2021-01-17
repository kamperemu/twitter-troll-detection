import pickle
from preprocesssklearn import *

# input data
n = int(input("no of sentences: "))
sentences = [str(input("enter sentence:")) for _ in range(n)]
# preprocessing
for i in range(len(sentences)):
    sentences[i] = removespchar(sentences[i])
    sentences[i] = stemmer(sentences[i])
    sentences[i] = removestopwords(sentences[i])
# loading model
tv = pickle.load(open("savedModel/nb/Tfidf.sav","rb"))
cv = pickle.load(open("savedModel/nb/bow.sav","rb"))
nbBow = pickle.load(open("savedModel/nb/bowmodel.sav", 'rb'))
nbTfidf = pickle.load(open("savedModel/nb/tfidfmodel.sav", 'rb'))
# prediction
tvsentences = tv.transform(sentences)
cvsentences = cv.transform(sentences)
print(nbBow.predict(cvsentences))
print(nbTfidf.predict(tvsentences))

