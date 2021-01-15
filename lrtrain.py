# imports
from preprocesssklearn import *
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
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
tweets['content']=tweets['content'].apply(denoise_text)
tweets['content']=tweets['content'].apply(remove_special_characters)
tweets['content']=tweets['content'].apply(simple_stemmer)
tweets['content']=tweets['content'].apply(remove_stopwords)


# encoding
tweets = tweets.sample(frac = 1)

xtrain = tweets.content[:int(round(4*(tweets['content'].size)/5))]
xtest = tweets.content[int(round(4*(tweets['content'].size)/5)):]

#labeling the sentient data
lb=LabelBinarizer()
labels=lb.fit_transform(tweets['label'])

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
    xtrain[i] = denoise_text(xtrain[i])
    xtrain[i] = remove_special_characters(xtrain[i])
    xtrain[i] = simple_stemmer(xtrain[i])
    xtrain[i] = remove_stopwords(xtrain[i])

for i in range(len(xtest)):
    xtest[i] = denoise_text(xtest[i])
    xtest[i] = remove_special_characters(xtest[i])
    xtest[i] = simple_stemmer(xtest[i])
    xtest[i] = remove_stopwords(xtest[i])

#labeling the sentient data
lb=LabelBinarizer()
ytrain=lb.fit_transform(ytrain)
ytest=lb.transform(ytest)




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
lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=42)
lr_bow=lr.fit(cvxtrain,ytrain)
lr_tfidf=lr.fit(tvxtrain,ytrain)

pred = lr_bow.predict(cvxtest)
print("Logistic regression Accuracy Score -> ",accuracy_score(pred, ytest)*100)
pred = lr_tfidf.predict(tvxtest)
print("Logistic regression Accuracy Score -> ",accuracy_score(pred, ytest)*100)

pickle.dump(lr_bow, open("savedModel/lr/bowmodel.sav","wb"))
pickle.dump(lr_tfidf, open("savedModel/lr/tfidfmodel.sav","wb"))
pickle.dump(tv, open("savedModel/lr/Tfidf.sav","wb"))
pickle.dump(cv, open("savedModel/lr/bow.sav","wb"))