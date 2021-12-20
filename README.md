# LSTM for Improved Classification of Encoded Objects
![Crocodile object classification](https://github.com/dtkutzke/lstm/blob/main/crocodile_output.gif)

## About
This project attempts to compare classification accuracy
for the Caltech 101 Silhouettes shapes using a long short-term
memory (LSTM) neural network and a single hidden layer perceptron.

## Requirements
The following are required to run
```
keras
tensorflow
scikit-learn
matplotlib
numpy
opencv-python
```

## Running
There are two primary files to run. (1) capture.py will run inference using a
mean-shift object tracker, specifying a region of interest with a click, then
it will perform inference using a pretrained model for both the MLP and the LSTM
learning algorithms. You can run this file ``out of the box" using
```
python capture.py crocodile_walking.mp4
```
Note that this utilizes the following configurations from training
Models/lstm_mlp_comparison_silhouettes94_MLP and Models/lstm_mlp_comparison_silhouettes94_LSTM to perform inference. These are pretrained models on four classes: crocodile, crocodile head, crab, and crayfish from the Caltech 101 dataset using augmented classes, with a total training size of 2,940 training images, 960 validation images, and 980 test images. A sequence size of 10 frames is required to perform inference with the LSTM.

To retrain, run (2) main.py. This file will retrain and save the new model. You can run this by doing
```
python main.py
```
Some important parameters are provided that you can change. First, you can the sequence length by changing the sequenceArray in the file. Second, you can also change the classes to train on by changing the classNamesSub array to include or exclude classes. Third, if you want to change how many augmented images to 
produce for a given class sample, you can change the 
augmentation_batch_size parameter. This will augment a single instance
20 times using the Keras.ImageGenerator with translation in both
horizontal and vertical directions, and rotation. Fourth, if you wish to change
(although I don't recommend it) the traing epochs, you can change epochsLstm andepochsMlp.

## General notes
* You need to have scikit-learn, keras, tensorflow, and opencv installed to 
run the examples and retrain. 
* The main files to understand are:
	- venv - this is the virtual environment file. I recommend setting
		 up a virtual environment of your own and placing these
                 files in there
	- venv/Data - this is the output from the tracker capture.py. Tracking
                      video will be output here with
                      "*_Classification_Video.mp4"
	- Output_Data - contains plots from training LSTM and MLP,
                        configuration parameter files "*_config.txt",
                        evaluation results "*_results.csv", example
                        augmented mask outputs "*_Object_Masks_<classname>.pdf"
	- Data - contains the dataset. Use caltech101_silhouettes_28_split1.mat
	- Models - saved models from the training runs. These can be used
                   for inference, once saved.
* Model saving can be toggled with the saveModel flag in main.py 
* You can change the Region of Interest (RoI) for the capture.py by changing
the regionWidth, and regionHeight parameters at the top of the file

