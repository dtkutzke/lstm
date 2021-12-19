from lstm import lstm
from mlp import mlp
from MaskDataUtils import DataSet
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
from PIL import Image
from keras.datasets import mnist
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.callbacks import History
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import tensorflow as tf
import scipy.io
from DataLogger import DataLogger

###########################
### DATA LOGGING SETUP ####
###########################
# Where to place any output data
dataDir = '../Output_Data'
# Prefix common to all output data
dataString = 'lstm_mlp_comparison_silhouettes'
# Should we save the output data
saveData = True
dl = DataLogger(dataDir, dataString)
runCount = dl.GetDataRunCount()
dl.dataString += runCount
print("*** Starting LSTM and MLP comparison session: ", dl.dataString)


########################################
### LOADING AND AUGMENTING THE DATA ####
########################################
# Load the data from the Caltech 101 Silhouettes
ds = DataSet()

classNamesSub = ['crab', 'crayfish', 'crocodile']
dl.classNames = classNamesSub
classNameIndices = ds.GetIndexForClassnames(classNamesSub)
nOutputsFromLSTM = np.max(classNameIndices)+1
dl.nOutputs = nOutputsFromLSTM

# Find each subset for the data
train_data_sub, train_labels_sub, val_data_sub, val_labels_sub, test_data_sub, test_labels_sub = ds.GetSubsetForClass(classNamesSub)

# Augment the data
augmentation_batch_size = 100
dl.dataAugSize = augmentation_batch_size
ds.AugmentData(augmentation_batch_size, train_data_sub, train_labels_sub, val_data_sub, val_labels_sub, test_data_sub, test_labels_sub)


# Now we training and test data
sequenceLength = 10
dl.lstmSeqLength = sequenceLength
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

dl.outputDims = nOutputsFromLSTM
dl.lstmInputDims = 784

epochsLstm = 20
dl.epochsLstm = epochsLstm
lstm = lstm(sequenceLength, 784, nOutputsFromLSTM, epochsLstm)

historyLstm = History()
lstm.fit(train_data_sub, train_labels_sub, val_data_sub, val_labels_sub, historyLstm)

epochsMlp = 5
dl.epochsMlp = epochsMlp
dl.mlpInputDims = 784
mlp = mlp(784, nOutputsFromLSTM, epochsMlp)

historyMlp = History()
# Reshape the data to make sure the MLP can see all of the data
#train_x = train_data_sub.reshape(train_data_sub.shape[0]*train_data_sub.shape[1], train_data_sub.shape[2])
#train_y = train_labels_sub.reshape(train_data_sub.shape[0]*train_data_sub.shape[1], train_labels_sub.shape[1])
mlp.fit(train_data_sub[:,0,:], train_labels_sub, val_data_sub[:,0,:], val_labels_sub, historyMlp)

N = 10
lstmAccuracy = np.zeros([N])
mlpAccuracy = np.zeros([N])
for i in range(N):
    lstmAccuracy[i] = lstm.predict(test_data_sub, test_labels_sub)
    mlpAccuracy[i] = mlp.predict(test_data_sub[:,0,:], test_labels_sub)

str1 = "LSTM mean classification accuracy after " + str(N)
str1 += " iterations: %.2f%%"
print(str1 % (np.mean(lstmAccuracy)*100))
str2 = "MLP mean classification accuracy after " + str(N)
str2 += " iterations: %.2f%%"
print(str2 % (np.mean(mlpAccuracy)*100))


fig = plt.figure(1)
plt.title('Loss vs epochs for LSTM')
plt.plot(np.arange(0,epochsLstm),historyLstm.history['val_loss'], 'k--', label='Validation loss')
plt.plot(np.arange(0,epochsLstm),historyLstm.history['loss'], 'r-', label='Training loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
#plt.show()
plt.savefig(dl.dataDir+"/"+dl.dataString+"_LSTM_Loss_vs_epochs.png")

fig = plt.figure(2)
plt.title('Loss vs epochs for MLP')
plt.plot(np.arange(0,epochsMlp),historyMlp.history['val_loss'], 'k--', label='Validation loss')
plt.plot(np.arange(0,epochsMlp),historyMlp.history['loss'], 'r-', label='Training loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
#plt.show()
plt.savefig(dl.dataDir+"/"+dl.dataString+"_MLP_Loss_vs_epochs.png")
#print(history.history['val_loss'])

if saveData:
    dl.SaveConfigDetails()