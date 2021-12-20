#!/bin/bash

if ! command -v virtualenv &> /dev/null
then
    echo "virtualenv could not be found"
    apt-get install virtual env ~
fi

virtualenv --no-site-packages venv

. venv/bin/activate


pip install keras
pip install tensorflow
pip install opencv-python
pip install scikit-learn
pip install matplotlib
pip install numpy
