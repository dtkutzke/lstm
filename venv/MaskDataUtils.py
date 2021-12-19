import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from numpy import expand_dims
import tensorflow as tf
import cv2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
#import pandas as pd
import itertools

'''Plotting utils for the image masks'''
class MaskPlotter:
    def __init__(self):
        pass

    def PlotMask(self, mask, title='Default'):
        plt.imshow(mask)
        plt.title(title)
        plt.show()

'''Load all of the data from the silhouettes'''
class DataSet:

    def __init__(self):
        self.mat = scipy.io.loadmat('../Data/caltech101_silhouettes_28_split1.mat')
        self.classLabels, self.xTrain, self.yTrain, self.xVal, self.yVal, self.xTest, self.yTest = self.LoadData()
        self.yTrain = self.yTrain.reshape((len(self.yTrain)))
        self.yVal = self.yVal.reshape((len(self.yVal)))
        self.yTest = self.yTest.reshape((len(self.yTest)))

    def GetDictValue(self, data_type):
        return self.mat[data_type]

    def UnpackData(self, data_type):
        tdFlat = self.GetDictValue(data_type)
        nSamples, _ = tdFlat.shape
        height = np.sqrt(flatDim).astype('int')
        data = np.zeros([nSamples, height, height])
        for i in range(nSamples):
            data[i, :, :] = tdFlat[i].reshape((height, height))

        return data

    def FlatTo3D(self, flat_in):
        width = flat_in.shape
        height = np.sqrt(width).astype('int')
        height = height[0]
        im = np.zeros([height, height, 3])
        for i in range(3):
            im[:,:,i] = flat_in.reshape((height, height))

        return im

    def LoadData(self):
        classLabels = self.GetDictValue('classnames')
        tmp = []
        _, nItems = classLabels.shape
        for i in range(nItems):
            tmp.append(classLabels[0][i][0])

        classLabels = tmp

        xTrain = self.GetDictValue('train_data')
        yTrain = self.GetDictValue('train_labels')

        xVal = self.GetDictValue('val_data')
        yVal = self.GetDictValue('val_labels')

        xTest = self.GetDictValue('test_data')
        yTest = self.GetDictValue('test_labels')

        return classLabels, xTrain, yTrain, xVal, yVal, xTest, yTest

    def GetIndexForClassnames(self, input_array):
        idx = []
        for i in range(len(input_array)):
            for j in range(len(self.classLabels)):
                if self.classLabels[j] == input_array[i]:
                    idx.append(j)
        return idx

    def GetSubsetForClass(self, class_names):
        idx = self.GetIndexForClassnames(class_names)
        trainSubset = []
        trainLabelsSubset = []
        valSubset = []
        valLabelsSubset = []
        testSubset = []
        testLabelsSubset = []
        cnt = 0
        for i in idx:
            mask = self.yTrain == i
            foundInstances = np.count_nonzero(np.array(mask))
            print('Found ', str(foundInstances), ' instances of class ', class_names[cnt], ' in the train data set')
            trainSubset.append(self.xTrain[mask])
            trainLabelsSubset.append(self.yTrain[mask])
            mask = self.yVal == i
            foundInstances = np.count_nonzero(np.array(mask))
            print('Found ', str(foundInstances), ' instances of class ', class_names[cnt], ' in the validation data set')
            valSubset.append(self.xVal[mask])
            valLabelsSubset.append(self.yVal[mask])
            mask = self.yTest == i
            foundInstances = np.count_nonzero(np.array(mask))
            print('Found ', str(foundInstances), ' instances of class ', class_names[cnt], ' in the test data set')
            testSubset.append(self.xTest[mask])
            testLabelsSubset.append(self.yTest[mask])
            cnt += 1

        return trainSubset, trainLabelsSubset, valSubset, valLabelsSubset, testSubset, testLabelsSubset

    def AugmentData(self, batch_size, train_data, train_labels, val_data=None, val_labels=None, test_data=None, test_labels=None):

        datagen = ImageDataGenerator(width_shift_range=[-5, 5], height_shift_range=[5, 5], rotation_range=45)

        # Handle the training data first
        for i in range(len(train_data)):
            cl = train_data[i]
            samples, flat_dim = cl.shape
            trainDataAug = np.zeros([samples*batch_size, flat_dim])
            trainLabelsAug = np.ones([samples*batch_size])
            cnt = 0
            for j in range(samples):
                converted = self.FlatTo3D(cl[j])
                it = datagen.flow(np.expand_dims(converted, 0), batch_size=1)
                for k in range(batch_size):
                    batch = it.next()
                    batch = batch[0]
                    # Threshold again so that everything is zeros or ones, since transformations change pixel
                    tmp = batch[:,:,0].flatten('C')
                    tmp[tmp<0.50] = 0.
                    tmp[tmp>0.50] = 1.
                    trainDataAug[cnt,:] = batch[:,:,0].flatten('C')

                    cnt += 1

            trainLabelsAug *= train_labels[i][0]

            train_data[i] = trainDataAug
            train_labels[i] = trainLabelsAug

        if val_data is not None:
            # Handle the training data first
            for i in range(len(val_data)):
                cl = val_data[i]
                samples, flat_dim = cl.shape
                valDataAug = np.zeros([samples * batch_size, flat_dim])
                valLabelsAug = np.ones([samples * batch_size])
                cnt = 0
                for j in range(samples):
                    # Generate 10 random augmentations
                    flat_dim = cl[j].shape
                    # converted = np.zeros([height, width, 3])
                    # for c in range(3):
                    #    converted[:,:,c] = cl[j]
                    converted = self.FlatTo3D(cl[j])
                    it = datagen.flow(np.expand_dims(converted, 0), batch_size=1)
                    for k in range(batch_size):
                        batch = it.next()
                        batch = batch[0]
                        valDataAug[cnt, :] = batch[:, :, 0].flatten('C')
                        cnt += 1

                valLabelsAug *= val_labels[i][0]

                val_data[i] = valDataAug
                val_labels[i] = valLabelsAug

        if test_data is not None:
            # Handle the training data first
            for i in range(len(test_data)):
                cl = test_data[i]
                samples, flat_dim = cl.shape
                testDataAug = np.zeros([samples * batch_size, flat_dim])
                testLabelsAug = np.ones([samples * batch_size])
                cnt = 0
                for j in range(samples):
                    # Generate 10 random augmentations
                    flat_dim = cl[j].shape
                    # converted = np.zeros([height, width, 3])
                    # for c in range(3):
                    #    converted[:,:,c] = cl[j]
                    converted = self.FlatTo3D(cl[j])
                    it = datagen.flow(np.expand_dims(converted, 0), batch_size=1)
                    for k in range(batch_size):
                        batch = it.next()
                        batch = batch[0]
                        testDataAug[cnt, :] = batch[:, :, 0].flatten('C')
                        cnt += 1

                testLabelsAug *= test_labels[i][0]

                test_data[i] = testDataAug
                test_labels[i] = testLabelsAug

    def MakeSequential(self, input, sequence_size):
        samples, features = input.shape
        if np.remainder(samples, sequence_size) == 0:
            reducedSize = int(samples/sequence_size)
            sequenceData = np.zeros([reducedSize, sequence_size, features])
            cnt = 0
            for i in range(reducedSize):
                for j in range(sequence_size):
                    sequenceData[i, j, :] = input[cnt]
                    cnt += 1

            return sequenceData

    def FromSubsetsPutTogether(self, subsets):
        total_size = 0
        for i in range(len(subsets)):
            total_size += len(subsets[i])

        if subsets[0].ndim == 3:
            _, sequence_length, features = subsets[0].shape
            cnt = 0
            newData = np.zeros([total_size, sequence_length, features])
            for i in range(len(subsets)):
                for j in range(len(subsets[i])):
                    newData[cnt, :, :] = subsets[i][j]
                    cnt += 1
            return newData
        else:
            cnt = 0
            newData = np.zeros([total_size])
            for i in range(len(subsets)):
                for j in range(len(subsets[i])):
                    newData[cnt] = subsets[i][j]
                    cnt += 1
            return newData


    def ConvertToSequenceDataset(self, subsets, sequence_size):
        for i in range(len(subsets)):
            sequenceData = self.MakeSequential(subsets[i], sequence_size)
            subsets[i] = sequenceData


