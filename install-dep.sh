#!/bin/bash

cd venv

virtualenv lstm

source lstm/bin/activate

pip install keras
pip install tensorflow
pip install opencv-python
pip install scikit-learn
pip install matplotlib
pip install numpy
