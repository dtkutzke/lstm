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

'''Creates a single hidden layer MLP for classification with the Caltech 101 Silhouettes'''
class mlp:
    def __init__(self, input_dim, output_dim, epochs=50, batch_size=10):
        self.inputDim = input_dim
        self.outputDim = output_dim
        self.epochs = epochs
        self.batchSize = batch_size
        self.model = Sequential()

    def fit(self, train_x, train_y, val_x, val_y, callback):
        # Define the first hidden layer as well
        self.model.add(Dense(100, input_dim=self.inputDim, activation='relu'))
        #self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(self.outputDim, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=self.epochs,
                  batch_size=self.batchSize, callbacks=[callback])

    def predict(self, test_x, test_y):
        scores = self.model.evaluate(test_x, test_y)
        print("MLP classification accuracy: %.2f%%" % (scores[1] * 100))
        return scores[1]

