import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow
import json
data=[]
with open('datasets/test.json','r') as json_file:
    data = json.load(json_file)

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
        
for i in range(len(xtrain)):
    print(ytrain[i], " ", xtrain[i])

for i in range(len(xtest)):
    print(ytest[i], " ", xtest[i])

