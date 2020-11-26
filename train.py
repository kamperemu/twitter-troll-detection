# removes default logs
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# imports required for the training algorithm
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle


# filtering based on training data and testing data
def filter_train(line):
    datatype = tf.strings.split(line,",", maxsplit=3)[1]
    return (True if datatype == 'train' else False)


def filter_test(line):
    datatype = tf.strings.split(line,",", maxsplit=3)[1]
    return (True if datatype == 'test' else False)


train = tf.data.TextLineDataset("datasets/test.csv").filter(filter_train)
test = tf.data.TextLineDataset("datasets/test.csv").filter(filter_test)

tweets = []
for line in train.skip(1).take(5):
    split_line = tf.strings.split(line, ",", maxsplit=3)
    tweet = split_line[2]
    tweet = tweet.numpy()
    tweets.append(tweet)





'''
tf_keras_tokenizer.fit_on_texts(text_data)
tf_keras_encoded = tf_keras_tokenizer.texts_to_sequences(text_data)
tf_keras_encoded = pad_sequences(tf_keras_encoded, padding="post") 

AUTOTUNE = tf.data.experimental.AUTOTUNE
train = train.map(encode_map_fn, num_parallel_calls=AUTOTUNE).cache()
train = train.shuffle(25000)
train = train.padded_batch(32, padded_shapes=([None], ()))

test = test.map(encode_map_fn)
test = test.padded_batch(32, padded_shapes=([None], ()))

model = keras.Sequential(
    [
        layers.Masking(mask_value=0),
        layers.Embedding(input_dim=len(vocabulary) + 2, output_dim=32,),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dense(1),
    ]
)

model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(3e-4, clipnorm=1),
    metrics=["accuracy"],
)

model.fit(train, epochs=15, verbose=2)
model.evaluate(test)
'''