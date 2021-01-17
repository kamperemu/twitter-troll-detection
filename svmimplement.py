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
tv = pickle.load(open("savedModel/svm/Tfidf.sav","rb"))
cv = pickle.load(open("savedModel/svm/bow.sav","rb"))
svmBow = pickle.load(open("savedModel/svm/bowmodel.sav", 'rb'))
svmTfidf = pickle.load(open("savedModel/svm/tfidfmodel.sav", 'rb'))
# prediction
tvsentences = tv.transform(sentences)
cvsentences = cv.transform(sentences)
print(svmBow.predict(cvsentences))
print(svmTfidf.predict(tvsentences))
