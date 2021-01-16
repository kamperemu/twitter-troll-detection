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
tv = pickle.load(open("savedModel/nb/Tfidf.sav","rb"))
cv = pickle.load(open("savedModel/nb/bow.sav","rb"))
nb_bow = pickle.load(open("savedModel/nb/bowmodel.sav", 'rb'))
nb_tfidf = pickle.load(open("savedModel/nb/tfidfmodel.sav", 'rb'))
# prediction
tvsentences = tv.transform(sentences)
cvsentences = cv.transform(sentences)
print(nb_bow.predict(cvsentences))
print(nb_tfidf.predict(tvsentences))
