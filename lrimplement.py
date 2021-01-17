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
tv = pickle.load(open("savedModel/lr/Tfidf.sav","rb"))
cv = pickle.load(open("savedModel/lr/bow.sav","rb"))
lrBow = pickle.load(open("savedModel/lr/bowmodel.sav", 'rb'))
lrTfidf = pickle.load(open("savedModel/lr/tfidfmodel.sav", 'rb'))
# prediction
tvsentences = tv.transform(sentences)
cvsentences = cv.transform(sentences)
print(lrBow.predict(cvsentences))
print(lrTfidf.predict(tvsentences))
