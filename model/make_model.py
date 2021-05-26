import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import config_tokenize

BASE = '/content/drive/My Drive/Abuse_filter'
os.chdir(BASE)


class My_Model(tf.keras.Model):
    def __init__(self, configs, BATCH_SIZE, EMBEDDING_DIM, UNIT):
        super(My_Model, self).__init__()
        self.batch_size = BATCH_SIZE
        self.unit = UNIT
        self.data_config = configs
        self.embedding = tf.keras.layers.Embedding(
            self.data_config['vocab_size'], EMBEDDING_DIM)
        self.layer1 = tf.keras.layers.Conv1D(128, kernel_size=1)
        self.layer2 = tf.keras.layers.Conv1D(128, kernel_size=3)
        self.layer3 = tf.keras.layers.Conv1D(128, kernel_size=5)
        self.maxpool = tf.keras.layers.GlobalMaxPooling1D()
        self.layer5 = tf.keras.layers.Dense(self.unit)
        self.layer6 = tf.keras.layers.Dense(self.unit)
        self.layer7 = tf.keras.layers.Dropout(0.2)
        self.last_layer = tf.keras.layers.Dense(2, activation=tf.nn.softmax)

    def call(self, x):
        emb_x = self.embedding(x)
        conv_1_out = self.layer1(emb_x)
        conv_3_out = self.layer2(emb_x)
        conv_5_out = self.layer3(emb_x)
        concat_shape = tf.concat([self.maxpool(conv_1_out), self.maxpool(
            conv_3_out), self.maxpool(conv_5_out)], axis=-1)

        result = self.last_layer(concat_shape)

        return result


def set_dyModel():
    if os.path.isfile(f'{BASE}/data/model_train.pickle'):
        with open(f'{BASE}/data/model_train.pickle', 'rb') as f:
            model_train = pickle.load(f)
    else:
        config_tokenize.make_tokenizer()
        with open(f'{BASE}/data/model_train.pickle', 'rb') as f:
            model_train = pickle.load(f)

    with open(f'{BASE}/configs/data_configs2.json') as f:
        configs = json.load(f)

    BATCH_SIZE = 1000
    embedding_dim = 128
    units = 250
    TEST_SPLIT = 0.2

    encoder = My_Model(configs, BATCH_SIZE, embedding_dim, units)

    encoder.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, optimizer='adam',
                    metrics=[tf.keras.metrics.sparse_categorical_accuracy])

    inputs = model_train['inputs']
    outputs = model_train['outputs']

    Input_train, Input_test, label_train, label_test = train_test_split(
        inputs, outputs, test_size=TEST_SPLIT, random_state=None)

    encoder.fit(Input_train, label_train, shuffle=True, batch_size=BATCH_SIZE, epochs=50, validation_data=(Input_test, label_test),
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')])

    encoder.save(f'{BASE}/configs/saved_model_ver2.h5py')


if __name__ == '__main__':
    set_dyModel()
