# Importing the required modules

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow
import json


# Making an array from the dataset file

data=[]
with open('datasets/test.json','r') as json_file:
    data = json.load(json_file)

# Classifying the training and testing, input and output of the dataset

xtrain = []
ytrain = []
xtest = []
ytest = []
sum0=0
sum1=0
for i in data:
    if i["label"] == 0:
        if sum0<=4000:
            xtrain.append(i["content"])
            ytrain.append(i["label"])
        else:
            xtest.append(i["content"])
            ytest.append(i["label"])
        sum0 += 1
    if i["label"] == 1:
        if sum1<=4000:
            xtrain.append(i["content"])
            ytrain.append(i["label"])
        else:
            xtest.append(i["content"])
            ytest.append(i["label"])
        sum1 += 1
     
     
# Converting the input layer to ASCII values.

xtrainord = []
for j in range(len(xtrain)):
    xtrainord.append([])
    for i in xtrain[j]:
        xtrainord[j].append(ord(i))
xtestord = []
for j in range(len(xtest)):
    xtestord.append([])
    for i in xtest[j]:
        xtestord[j].append(ord(i))

print(xtrainord)
print(xtestord)
