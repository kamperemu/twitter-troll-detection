import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow
import json

with open('test.json','r') as json_file:
    data = json.load(json_file)
    print(len(data))