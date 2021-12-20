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

