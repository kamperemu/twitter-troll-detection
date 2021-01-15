# removes unnecessary logs
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# imports required for the training algorithm
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import pickle
import random

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

# tokenization go brrr
tokenizer = Tokenizer(num_words=vocabSize, oov_token=oov)
tokenizer.fit_on_texts(xtrain)

wordIndex = tokenizer.word_index

# preparing training data for neural network
xtrainencoded = tokenizer.texts_to_sequences(xtrain)
xtrainpadded = pad_sequences(xtrainencoded, maxlen=maxInput, padding=padType, truncating=truncType)
xtrain = np.asarray(xtrainpadded).astype(np.float32)
ytrain = np.asarray(ytrain).astype(np.float32)

# preparing testing data for neural network
xtestencoded = tokenizer.texts_to_sequences(xtest)
xtestpadded = pad_sequences(xtestencoded, maxlen=maxInput, padding=padType, truncating=truncType)
xtest = np.asarray(xtestpadded).astype(np.float32)
ytest = np.asarray(ytest).astype(np.float32)

# neural network
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocabSize, outputDim, input_length=maxInput),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
numEpochs = 30
history = model.fit(xtrain, ytrain, epochs=numEpochs, validation_data=(xtest, ytest), verbose=2)
#model.summary()
'''
# graphs
import matplotlib.pyplot as plt

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
'''
# saving the tokenizer
with open('savedModel/nn/basic/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# saving trained model
model.save("savedModel/nn/basic/model")
