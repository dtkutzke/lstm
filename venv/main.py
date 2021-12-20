from lstm import lstm
from mlp import mlp
from MaskDataUtils import DataSet
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
from PIL import Image
from sklearn.utils import shuffle
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
# Should we save the output data
saveData = True
# Should we save the keras model?
saveModel = True
lstmAvgAccuracy = []
lstmStdAccuracy = []
mlpAvgAccuracy = []
mlpStdAccuracy = []
sequenceArray = [10]
for sequenceLength in sequenceArray:
    # Prefix common to all output data
    dataString = 'lstm_mlp_comparison_silhouettes'
    dl = DataLogger(dataDir, dataString)
    runCount = dl.GetDataRunCount()
    dl.dataString += runCount
    print("*** Starting LSTM and MLP comparison session: ", dl.dataString)
    print("*** Sequence length: ", sequenceLength)


    ########################################
    ### LOADING AND AUGMENTING THE DATA ####
    ########################################
    # Load the data from the Caltech 101 Silhouettes
    ds = DataSet()

    classNamesSub = ['crab', 'crayfish', 'crocodile', 'crocodile head']
    dl.classNames = classNamesSub
    classNameIndices = ds.GetIndexForClassnames(classNamesSub)
    nOutputsFromLSTM = np.max(classNameIndices)+2
    dl.nOutputs = nOutputsFromLSTM

    # Find each subset for the data
    train_data_sub, train_labels_sub, val_data_sub, val_labels_sub, test_data_sub, test_labels_sub = ds.GetSubsetForClass(classNamesSub)

    # Augment the data
    augmentation_batch_size = 20
    dl.dataAugSize = augmentation_batch_size
    ds.AugmentData(augmentation_batch_size, train_data_sub, train_labels_sub, val_data_sub, val_labels_sub, test_data_sub, test_labels_sub)


    # Now we training and test data

    #sequenceLength = i
    dl.lstmSeqLength = sequenceLength
    ds.ConvertToSequenceDataset(train_data_sub, sequenceLength)
    train_data_sub = ds.FromSubsetsPutTogether(train_data_sub)
    dl.trainSize = len(train_data_sub)
    train_labels_sub = ds.FromSubsetsPutTogether(train_labels_sub)
#    train_labels_sub = to_categorical(train_labels_sub[::sequenceLength].astype('int'))
    train_labels_sub = to_categorical(train_labels_sub.astype('int'))
    ds.ConvertToSequenceDataset(val_data_sub, sequenceLength)
    val_data_sub = ds.FromSubsetsPutTogether(val_data_sub)
    dl.valSize = len(val_data_sub)
    val_labels_sub = ds.FromSubsetsPutTogether(val_labels_sub)
    val_labels_sub = to_categorical(val_labels_sub.astype('int'))

    ds.ConvertToSequenceDataset(test_data_sub, sequenceLength)
    test_data_sub = ds.FromSubsetsPutTogether(test_data_sub)
    dl.testSize = len(test_data_sub)
    test_labels_sub = ds.FromSubsetsPutTogether(test_labels_sub)
    test_labels_sub = to_categorical(test_labels_sub.astype('int'))

    #########################################
    ### CREATING THE LSTM AND MLP OBJECTS ###
    #########################################
    dl.outputDims = nOutputsFromLSTM
    dl.lstmInputDims = 784

    epochsLstm = 15
    dl.epochsLstm = epochsLstm
    lstm_ = lstm(sequenceLength, 784, nOutputsFromLSTM, epochsLstm)

    historyLstm = History()
    lstm_.fit(train_data_sub, train_labels_sub, val_data_sub, val_labels_sub, historyLstm)

    epochsMlp = 5
    dl.epochsMlp = epochsMlp
    dl.mlpInputDims = 784
    mlp_ = mlp(784, nOutputsFromLSTM, epochsMlp)

    historyMlp = History()
    # Reshape the data to make sure the MLP can see all of the data
    #train_x = train_data_sub.reshape(train_data_sub.shape[0]*train_data_sub.shape[1], train_data_sub.shape[2])
    #train_y = train_labels_sub.reshape(train_data_sub.shape[0]*train_data_sub.shape[1], train_labels_sub.shape[1])
    mlp_.fit(train_data_sub[:,0,:], train_labels_sub, val_data_sub[:,0,:], val_labels_sub, historyMlp)

    if saveModel:
        lstm_.model.save("../Models/"+dl.dataString+"_LSTM")
        mlp_.model.save("../Models/"+dl.dataString+"_MLP")
    #####################################################
    ### FIT THE MODELS OVER SEVERAL ITER THEN AVERAGE ###
    #####################################################
    N = 30
    dl.accIter = N
    lstmAccuracy = np.zeros([N])
    mlpAccuracy = np.zeros([N])
    for i in range(N):
        test_x, test_y = shuffle(test_data_sub, test_labels_sub)
        lstmAccuracy[i] = lstm_.evaluate(test_x, test_y)
        mlpAccuracy[i] = mlp_.evaluate(test_x[:,0,:], test_y)

    str1 = "LSTM mean classification accuracy after " + str(N)
    str1 += " iterations: %.2f%%"
    print(str1 % (np.mean(lstmAccuracy)*100))
    str2 = "MLP mean classification accuracy after " + str(N)
    str2 += " iterations: %.2f%%"
    print(str2 % (np.mean(mlpAccuracy)*100))

    lstmAvgAccuracy.append(np.mean(lstmAccuracy)*100)
    lstmStdAccuracy.append(np.std(lstmAccuracy))
    mlpAvgAccuracy.append(np.mean(mlpAccuracy)*100)
    mlpStdAccuracy.append(np.std(mlpAccuracy))

    ##########################
    ### PLOT LOSS FIGURES ####
    ##########################
    fig = plt.figure(1)
    #plt.title('Loss vs epochs for LSTM')
    plt.plot(np.arange(0,epochsLstm),historyLstm.history['val_loss'], 'k--', label='Validation loss')
    plt.plot(np.arange(0,epochsLstm),historyLstm.history['loss'], 'r-', label='Training loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    #plt.show()
    plt.savefig(dl.dataDir+"/"+dl.dataString+"_LSTM_Loss_vs_epochs.pdf")
    plt.close(fig)

    fig = plt.figure(2)
    #plt.title('Loss vs epochs for MLP')
    plt.plot(np.arange(0,epochsMlp),historyMlp.history['val_loss'], 'k--', label='Validation loss')
    plt.plot(np.arange(0,epochsMlp),historyMlp.history['loss'], 'r-', label='Training loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    #plt.show()
    plt.savefig(dl.dataDir+"/"+dl.dataString+"_MLP_Loss_vs_epochs.pdf")
    plt.close(fig)
    #print(history.history['val_loss'])

    ######################################
    ### PLOT EXAMPLE SEQUENCE FIGURES ####
    ######################################
    if sequenceLength <= 10:
        seqStarts = []
        for i in range(len(classNameIndices)):
            for j in range(len(train_labels_sub)):
                if np.argmax(train_labels_sub[j])-1==classNameIndices[i]:
                    seqStarts.append(j)
                    break

        for s in range(len(seqStarts)):
            fig = plt.figure(3)
            plt.title('Encoded object sequence for ' + classNamesSub[s])
            for i in range(sequenceLength):
                if sequenceLength > 5:
                    plt.subplot(2, 5, 1+i)
                else:
                    plt.subplot(1, 5, 1+i)

                plt.imshow(ds.FlatTo3D(train_data_sub[seqStarts[s],i,:]))
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)

            plt.savefig(dl.dataDir+"/"+dl.dataString+"_Object_Masks_"+classNamesSub[s]+".pdf")
            plt.close(fig)

    if saveData:
        dl.SaveConfigDetails()
        dl.SaveResults(np.mean(lstmAccuracy)*100, np.mean(mlpAccuracy)*100)


fig = plt.figure(120)
plt.errorbar(sequenceArray,lstmAvgAccuracy,yerr=lstmStdAccuracy,fmt='k-',label='LSTM classification accuracy')
plt.errorbar(sequenceArray,mlpAvgAccuracy,yerr=mlpStdAccuracy,fmt='k--', label='MLP classification accuracy')
plt.xlabel('Input sequence length')
plt.ylabel('Classification accuracy (%)')
plt.legend()
plt.savefig(dl.dataDir+"/"+dl.dataString+"_Classification_Accuracy.pdf")
plt.close(fig)