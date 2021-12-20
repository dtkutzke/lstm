import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
from PIL import Image
from keras.datasets import mnist
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import tensorflow as tf
import scipy.io
from MaskDataUtils import DataSet, MaskPlotter

'''LSTM implementation for evaluation
Assumes that this is for classification'''
class lstm:
    def __init__(self, sequence_length, input_dim, output_length, epochs=5, batch_size=64):
        self.sequenceLength = sequence_length
        self.inputDim = input_dim
        self.outputLength = output_length
        self.epochs = epochs
        self.batchSize = batch_size
        self.model = Sequential()

    def fit(self, train_x, train_y, val_x, val_y, call_back):
        self.model.add(LSTM(30, input_shape=(self.sequenceLength, self.inputDim)))
        self.model.add(Dropout(0.5))
        #self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(self.outputLength, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())
        self.model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=self.epochs, batch_size=self.batchSize, callbacks=[call_back])

    def evaluate(self, test_x, test_y):
        scores = self.model.evaluate(test_x, test_y, verbose=0)
        print("LSTM classification accuracy: %.2f%%" % (scores[1] * 100))
        return scores[1]

    def predict(self, test_instance):
        prediction = self.model.predict(test_instance)
        return prediction

    def loadModel(self, path):
        self.model = keras.models.load_model(path)
