# removes default logs
# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# imports required for the training algorithm
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json

# predefined variables
vocab_size = 10000
embedding_dim = 16
max_length = 200
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"


# training data and testing data
with open("datasets/test.json", 'r') as f:
    datastore = json.load(f)

xtrain = []
ytrain = []
xtest = []
ytest = []

for item in datastore:
    if item["type"] == "train":
        xtrain.append(item['content'])
        ytrain.append(item['label'])
    if item["type"] == "test":
        xtest.append(item['content'])
        ytest.append(item['label'])


# Tokenization go brrr
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(xtrain)

word_index = tokenizer.word_index

# preparing training data for neural network
xtrainencoded = tokenizer.texts_to_sequences(xtrain)
xtrainpadded = pad_sequences(xtrainencoded, maxlen=max_length, padding=padding_type, truncating=trunc_type)
xtrain = np.asarray(xtrainpadded).astype(np.float32)
ytrain = np.asarray(ytrain).astype(np.float32)

# preparing testing data for neural network
xtestencoded = tokenizer.texts_to_sequences(xtest)
xtestpadded = pad_sequences(xtestencoded, maxlen=max_length, padding=padding_type, truncating=trunc_type)
xtest = np.asarray(xtestpadded).astype(np.float32)
ytest = np.asarray(ytest).astype(np.float32)

# neural network
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
num_epochs = 30
history = model.fit(xtrain, ytrain, epochs=num_epochs, validation_data=(xtest, ytest), verbose=2)


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