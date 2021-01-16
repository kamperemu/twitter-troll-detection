import pickle
from preprocesssklearn import *

# input data
n = int(input("no of sentences: "))
sentences = [str(input("enter sentence:")) for _ in range(n)]
# preprocessing
for i in range(len(sentences)):
    sentences[i] = remove_special_characters(sentences[i])
    sentences[i] = simple_stemmer(sentences[i])
    sentences[i] = remove_stopwords(sentences[i])
# loading model
tv = pickle.load(open("savedModel/svm/Tfidf.sav","rb"))
cv = pickle.load(open("savedModel/svm/bow.sav","rb"))
svm_bow = pickle.load(open("savedModel/svm/bowmodel.sav", 'rb'))
svm_tfidf = pickle.load(open("savedModel/svm/tfidfmodel.sav", 'rb'))
# prediction
tvsentences = tv.transform(sentences)
cvsentences = cv.transform(sentences)
print(svm_bow.predict(cvsentences))
print(svm_tfidf.predict(tvsentences))
