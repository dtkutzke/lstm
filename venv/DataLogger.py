import os.path
import os
import sys
from datetime import datetime

class DataLogger:
    def __init__(self, data_dir, data_string):
        self.dataDir = data_dir
        self.dataString = data_string
        self.lstmInputDims = 0
        self.mlpInputDims = 0
        self.classNames = []
        self.outputDims = 0
        self.dataAugSize = 0
        self.lstmSeqLength = 0
        self.epochsLstm = 0
        self.epochsMlp = 0
        self.trainSize = 0
        self.valSize = 0
        self.testSize = 0
        self.accIter = 0

    '''Save a small file containing the configuration of this run'''
    def SaveConfigDetails(self):
        with open(self.dataDir + "/" + self.dataString + "_config.txt", 'a', encoding='utf-8') as f:
            f.write("*** CONFIGURATION FILE *** " + self.dataString + "\n")
            f.write("Run date/time: " + datetime.now().strftime("%m/%d/%y %H:%M:%S") + "\n")
            f.write("LSTM feature size: " + str(self.lstmInputDims) + "\n")
            f.write("MLP feature size: " + str(self.mlpInputDims) + "\n")
            f.write("Class names explored: " + str(self.classNames) + "\n")
            f.write("Output dimensions (one-hot): " + str(self.outputDims) + "\n")
            f.write("Data augmentation size (how many to generate per frame): " + str(self.dataAugSize) + "\n")
            f.write("LSTM 'sequence length' from (samples, 'sequence length', features): " + str(self.lstmSeqLength) + "\n")
            f.write("Epochs for LSTM: " + str(self.epochsLstm) + "\n")
            f.write("Epochs for MLP: " + str(self.epochsMlp) + "\n")
            f.write("Training dataset size: " + str(self.trainSize) + "\n")
            f.write("Validation dataset size: " + str(self.valSize) + "\n")
            f.write("Test dataset size: " + str(self.testSize) + "\n")
            f.write("Accuracy iterations: " + str(self.accIter) + "\n")
            # f.write("Mean pixel bin frequency index (bgr): (" + str(b) + "," + str(g) + "," + str(r) + ")\n")
            # f.write("Figure output boolean: " + str(saveData) + "\n")


    '''Helper function to extract the iterator for this data run'''
    def GetDataRunCount(self):
        if os.path.isdir(self.dataDir):
            # Load the files, extract iter from last file
            onlyfiles = [f for f in os.listdir(self.dataDir) if os.path.isfile(os.path.join(self.dataDir, f))]
            if onlyfiles:
                onlyfiles.sort()
                last_file = onlyfiles[-1]
                if last_file.find(self.dataString) == 0:
                    spl = last_file.split(self.dataString, 1)[1]
                    currIt = spl.split('_', 1)[0]
                    if int(currIt) < 9:
                        return "0" + str(int(currIt) + 1)
                    else:
                        return str(int(currIt) + 1)
            else:
                return "0" + str(1)
        else:
            print("Creating data directory to save results")
            os.mkdir(self.dataDir)
            return "0" + str(1)

    def SaveResults(self, lstm_results, ml_results):
       with open(self.dataDir + "/" + self.dataString + "_results.csv", 'a', encoding='utf-8') as f:
           f.write("LSTM, MLP\n")
           f.write(str(lstm_results)+","+str(ml_results)+"\n")