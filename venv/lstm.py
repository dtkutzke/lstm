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

# Load the data from the Caltech 101 Silhouettes
ds = DataSet()

# Create a subset of augmented data
# Note that with two class, we can still use binary cross entropy loss in LSTM
classNamesSub = ['ceiling fan', 'crab', 'crocodile head']
classNameIndices = ds.GetIndexForClassnames(classNamesSub)
nOutputsFromLSTM = np.max(classNameIndices)+1
train_data_sub, train_labels_sub, val_data_sub, val_labels_sub, test_data_sub, test_labels_sub = ds.GetSubsetForClass(classNamesSub)

# Augment the data
augmentation_batch_size = 100
ds.AugmentData(augmentation_batch_size, train_data_sub, train_labels_sub, val_data_sub, val_labels_sub, test_data_sub, test_labels_sub)

# Now we training and test data
sequenceLength = 10

ds.ConvertToSequenceDataset(train_data_sub, sequenceLength)
train_data_sub = ds.FromSubsetsPutTogether(train_data_sub)
train_labels_sub = ds.FromSubsetsPutTogether(train_labels_sub)
train_labels_sub = to_categorical(train_labels_sub[::sequenceLength].astype('int'))

ds.ConvertToSequenceDataset(val_data_sub, sequenceLength)
val_data_sub = ds.FromSubsetsPutTogether(val_data_sub)
val_labels_sub = ds.FromSubsetsPutTogether(val_labels_sub)
val_labels_sub = to_categorical(val_labels_sub[::sequenceLength].astype('int'))

ds.ConvertToSequenceDataset(test_data_sub, sequenceLength)
test_data_sub = ds.FromSubsetsPutTogether(test_data_sub)
test_labels_sub = ds.FromSubsetsPutTogether(test_labels_sub)
test_labels_sub = to_categorical(test_labels_sub[::sequenceLength].astype('int'))


# Now train the LSTM
embedding_vector_length = 784
model = Sequential()
model.add(LSTM(100, input_shape=(sequenceLength, embedding_vector_length)))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(nOutputsFromLSTM, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(train_data_sub, train_labels_sub, validation_data=(val_data_sub, val_labels_sub), epochs=5, batch_size=64)
scores = model.evaluate(test_data_sub, test_labels_sub, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Now we clean up everything
idx = np.argmax(test_labels_sub[0])
suptitle = 'Sequence data for ' + ds.classLabels[idx]
plt.suptitle(suptitle)
for i in range(sequenceLength):
 # define subplot
    if sequenceLength < 5:
        plt.subplot(150+1+i)
    else:
        plt.subplot(2, 5, 0+1+i)
    # generate batch of images
    # plot raw pixel data
    plt.imshow(ds.FlatTo3D(train_data_sub[0][i]))
# show the figure
plt.show()
