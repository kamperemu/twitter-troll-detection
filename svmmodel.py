# imports
from preprocesssklearn import *
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
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
    xtrain[i] = removestopwords(xtrain[i])

for i in range(len(xtest)):
    xtest[i] = removespchar(xtest[i])
    xtest[i] = stemmer(xtest[i])
    xtest[i] = removestopwords(xtest[i])
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
#svm=SVC(kernel='linear',C=10,gamma=100,cache_size=10000)
svm=LinearSVC()
#svmBow=svm.fit(cvxtrain,ytrain)
svmTfidf=svm.fit(tvxtrain,ytrain)
print("model trained")
print()
print()
print()
print()
#pred = svmBow.predict(cvxtest)
#print("Support Vector Machine Accuracy Score -> ",accuracy_score(pred, ytest)*100)
pred = svmTfidf.predict(tvxtest)
print("Support Vector Machine Accuracy Score -> ",accuracy_score(pred, ytest)*100)
print()
print()
print()
print()
print("saving the encoder and model")
#pickle.dump(svmBow, open("savedModel/svm/bowmodel.sav","wb"))
pickle.dump(svmTfidf, open("savedModel/svm/tfidfmodel.sav","wb"))
pickle.dump(tv, open("savedModel/svm/Tfidf.sav","wb"))
#pickle.dump(cv, open("savedModel/svm/bow.sav","wb"))
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
