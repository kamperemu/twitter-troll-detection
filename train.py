import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dataset = tf.data.TextLineDataset("datasets/test.csv")

for line in dataset.skip(1).take(20001):
    print(line)