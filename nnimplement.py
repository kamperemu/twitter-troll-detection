# removes unnecessary logs
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# importing necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# predefined variables
vocabSize = 10000
outputDim = 16
maxInput = 200
truncType='post'
padType='post'
oov = "<OOV>"

# loading the tokenizer and model
with open('savedModel/nn/basic/tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

model = tf.keras.models.load_model("savedModel/nn/basic/model")

n = int(input("no of sentences: "))
sentences = [input("Enter sentence:") for _ in range(n)]
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=maxInput, padding=padType, truncating=truncType)
print(model.predict(padded))