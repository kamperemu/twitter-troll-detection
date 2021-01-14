import pickle
from preprocesssklearn import *

# input data
n = int(input("no of sentences: "))
sentences = [str(input("enter sentence:")) for _ in range(n)]
# preprocessing
for i in range(len(sentences)):
    sentences[i] = denoise_text(sentences[i])
    sentences[i] = remove_special_characters(sentences[i])
    sentences[i] = simple_stemmer(sentences[i])
    sentences[i] = remove_stopwords(sentences[i])
# loading model
tv = pickle.load(open("savedModel/lr/Tfidf.sav","rb"))
cv = pickle.load(open("savedModel/lr/bow.sav","rb"))
lr_bow = pickle.load(open("savedModel/lr/bowmodel.sav", 'rb'))
lr_tfidf = pickle.load(open("savedModel/lr/tfidfmodel.sav", 'rb'))
# prediction
tvsentences = tv.transform(sentences)
cvsentences = cv.transform(sentences)
print(lr_bow.predict(cvsentences))
print(lr_tfidf.predict(tvsentences))
