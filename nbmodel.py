# imports
from preprocesssklearn import *
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
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
import json
import random
# training data and testing data
with open("datasets/test.json", 'r') as f:
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


# preprocessing
for i in range(len(xtrain)):
    xtrain[i] = removespchar(xtrain[i])
    xtrain[i] = stemmer(xtrain[i])
    xtrain[i] = removestopwords(xtrain[i])

for i in range(len(xtest)):
    xtest[i] = removespchar(xtest[i])
    xtest[i] = stemmer(xtest[i])
    xtest[i] = removestopwords(xtest[i])




# common for json and csv


#Count vectorizer for bag of words
cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))
cvxtrain=cv.fit_transform(xtrain)
cvxtest=cv.transform(xtest)

#Tfidf vectorizer
tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))
tvxtrain=tv.fit_transform(xtrain)
tvxtest=tv.transform(xtest)

#training the model
nb=naive_bayes.MultinomialNB()
nbBow=nb.fit(cvxtrain,ytrain)
nbTfidf=nb.fit(tvxtrain,ytrain)

pred = nbBow.predict(cvxtest)
print("Naive Bayes Accuracy Score -> ",accuracy_score(pred, ytest)*100)
pred = nbTfidf.predict(tvxtest)
print("Naive Bayes Accuracy Score -> ",accuracy_score(pred, ytest)*100)

pickle.dump(nbBow, open("savedModel/nb/bowmodel.sav","wb"))
pickle.dump(nbTfidf, open("savedModel/nb/tfidfmodel.sav","wb"))
pickle.dump(tv, open("savedModel/nb/Tfidf.sav","wb"))
pickle.dump(cv, open("savedModel/nb/bow.sav","wb"))