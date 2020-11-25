import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def filter_train(line):
    split_line = tf.strings.split(line,",", maxsplit=3)
    dataset_belonging = split_line[1]
    
    return (True if dataset_belonging == 'train' else False)


def filter_test(line):
    split_line = tf.strings.split(line,",", maxsplit=3)
    dataset_belonging = split_line[1]
    
    return (True if dataset_belonging == 'test' else False)


train = tf.data.TextLineDataset("datasets/test.csv").filter(filter_train)
test = tf.data.TextLineDataset("datasets/test.csv").filter(filter_test)


for line in test.skip(1).take(5):
    print(tf.strings.split(line, ",", maxsplit=3))