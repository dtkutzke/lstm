# lstm

## About
This project attempts to compare classification accuracy
for the Caltech 101 Silhouettes shapes using a long short-term
memory (LSTM) neural network and a single hidden layer perceptron.

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
Some important parameters are provided that you can change. First, you can the sequence length by changing the sequenceArray in the file. You can also change the classes to train on by changing the classNamesSub array to include or exclude classes. 
