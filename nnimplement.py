# removes unnecessary logs
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# importing necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# predefined variables
maxInput = 200
truncType='post'
padType='post'

# loading the tokenizer and model
with open('savedModel/nn/basic/tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

model = tf.keras.models.load_model("savedModel/nn/basic/model")

n = int(input("no of sentences: "))
sentences = [input("Enter sentence:") for _ in range(n)]
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=maxInput, padding=padType, truncating=truncType)
print(tf.round(model.predict(padded)))

