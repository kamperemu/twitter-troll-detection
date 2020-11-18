import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow
import json
data=[]
with open('datasets/test.json','r') as json_file:
    data = json.load(json_file)

sum0 = 0
sum1 = 0
for i in data:
    if i["label"]==0:
        sum0+=1
    if i["label"]==1:
        sum1+=1

print(sum0)
print(sum1)
print(sum0+sum1)