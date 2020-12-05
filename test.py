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
with open('tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

model = tf.keras.models.load_model("model")

sentence = [input()]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=maxInput, padding=padType, truncating=truncType)
print(model.predict(padded))