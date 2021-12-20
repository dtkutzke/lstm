#!/bin/bash

if ! command -v virtualenv &> /dev/null
then
    echo "<the_command> could not be found"
    apt-get install virtual env ~
fi

virtualenv venv

source venv/bin/activate

pip install keras
pip install tensorflow
pip install opencv-python
pip install scikit-learn
pip install matplotlib
pip install numpy
