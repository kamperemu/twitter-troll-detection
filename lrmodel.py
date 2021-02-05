# imports
from preprocesssklearn import *
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

'''
# csv algo
import pandas as pd
tweets = pd.read_csv("datasets/test.csv")

# preprocessing
tweets['content']=tweets['content'].apply(str)
tweets['content']=tweets['content'].apply(removespchar)
tweets['content']=tweets['content'].apply(stemmer)
tweets['content']=tweets['content'].apply(removestopwords)


# encoding
tweets = tweets.sample(frac = 1)

xtrain = tweets.content[:int(round(4*(tweets['content'].size)/5))]
xtest = tweets.content[int(round(4*(tweets['content'].size)/5)):]
ytrain = labels[:int(round(4*(labels.size)/5))]
ytest = labels[int(round(4*(labels.size)/5)):]
'''
# json algo
print("loading dataset")
import json
import random
# training data and testing data
with open("datasets/data.json", 'r') as f:
    tweets = json.load(f)
random.shuffle(tweets)
train = tweets[:int(round(4*len(tweets)/5))]
test = tweets[int(round(4*len(tweets)/5)):len(tweets)]
xtrain = []
ytrain = []
xtest = []
ytest = []

for tweet in train:
    xtrain.append(tweet['content'])
    ytrain.append(tweet['label'])
    
for tweet in test:
    xtest.append(tweet['content'])
    ytest.append(tweet['label'])

print("dataset loaded")
print()
print()
print()
print()
print("preprocessing data")
# preprocessing
for i in range(len(xtrain)):
    xtrain[i] = removespchar(xtrain[i])
    xtrain[i] = stemmer(xtrain[i])

for i in range(len(xtest)):
    xtest[i] = removespchar(xtest[i])
    xtest[i] = stemmer(xtest[i])
    
print("data preprocessed")
print()
print()
print()
print()


# common for json and csv
print("encoding data")
'''
#Count vectorizer for bag of words
cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))
cvxtrain=cv.fit_transform(xtrain)
cvxtest=cv.transform(xtest)
'''
#Tfidf vectorizer
tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))
tvxtrain=tv.fit_transform(xtrain)
tvxtest=tv.transform(xtest)
print("data encoded")
print()
print()
print()
print()

print("training the model")
#training the model
lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=42)
# lrBow=lr.fit(cvxtrain,ytrain)
lrTfidf=lr.fit(tvxtrain,ytrain)
print("model trained")
print()
print()
print()
print()
# pred = lrBow.predict(cvxtest)
# print("Logistic regression Accuracy Score -> ",accuracy_score(pred, ytest)*100)
pred = lrTfidf.predict(tvxtest)
print("Logistic regression Accuracy Score -> ",accuracy_score(pred, ytest)*100)
print()
print()
print()
print()
print("saving the encoder and model")
# pickle.dump(lrBow, open("savedModel/lr/bowmodel.sav","wb"))
pickle.dump(lrTfidf, open("savedModel/lr/tfidfmodel.sav","wb"))
pickle.dump(tv, open("savedModel/lr/Tfidf.sav","wb"))
# pickle.dump(cv, open("savedModel/lr/bow.sav","wb"))
print("encoder and model saved")
'''
# graphs
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

array = confusion_matrix(ytest,pred,labels=[1,0])
df_cm = pd.DataFrame(array, range(2), range(2))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

plt.show()
'''