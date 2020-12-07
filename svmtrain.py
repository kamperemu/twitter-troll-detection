# removes unnecessary logs
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# imports required for the training algorithm
from sklearn import svm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json


# predefined variables
vocabSize = 10000
outputDim = 16
maxInput = 200
truncType='post'
padType='post'
oov = "<OOV>"


# training data and testing data
with open("datasets/test.json", 'r') as f:
    tweets = json.load(f)

xtrain = []
ytrain = []
xtest = []
ytest = []

for tweet in tweets:
    if tweet["type"] == "train":
        xtrain.append(tweet['content'])
        ytrain.append(tweet['label'])
    if tweet["type"] == "test":
        xtest.append(tweet['content'])
        ytest.append(tweet['label'])


# tokenization go brrr
tokenizer = Tokenizer(num_words=vocabSize, oov_token=oov)
tokenizer.fit_on_texts(xtrain)

wordIndex = tokenizer.word_index


# preparing training data for neural network
xtrainencoded = tokenizer.texts_to_sequences(xtrain)
xtrainpadded = pad_sequences(xtrainencoded, maxlen=maxInput, padding=padType, truncating=truncType)
xtrain = np.asarray(xtrainpadded).astype(np.float32)

# preparing testing data for neural network
xtestencoded = tokenizer.texts_to_sequences(xtest)
xtestpadded = pad_sequences(xtestencoded, maxlen=maxInput, padding=padType, truncating=truncType)
xtest = np.asarray(xtestpadded).astype(np.float32)


# SVM

clf = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
clf.fit(xtrain, ytrain)
pred = clf.predict(xtest)
sum = 0
for i in range(len(ytest)):
    if ytest[i] == pred[i]:
        sum+=1

acc = (sum/len(ytest))*100
print(acc)
'''
# fit the training dataset on the NB classifier
clf = naive_bayes.MultinomialNB()
clf.fit(xtrain,ytrain)
pred = clf.predict(xtest)
sum = 0
for i in range(len(ytest)):
    if ytest[i] == pred[i]:
        sum+=1

acc = (sum/len(ytest))*100
print(acc)
'''