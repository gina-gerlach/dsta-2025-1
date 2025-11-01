# dsta-2025-1
Datascience Toolkits and Architectures

Gina Gerlach

This project trains a Convolutional Neural Network (CNN) to recognize handwritten digits (0–9) from the MNIST dataset.  
The model achieves around 99% test accuracy after training.

## Overview

The MNIST dataset consists of 70,000 grayscale images of handwritten digits (28×28 pixels each).  

This code:

1. Loads and preprocesses the dataset

2. Builds a CNN with Keras

3. Trains the model on the training data

4. Evaluates its performance on the test data


## Requirements

Make sure you have the following installed:

- Python 3.10  and -pip -venv

- Git

> ```sudo apt install -y python3 python3-pip git python3-venv```

- Keras

- TensorFlow

- NumPy

> ```pip install tensorflow keras numpy```

## Run the code

To run follow the process below:

1. Clone the repository

> ```git clone git@github.com:gina-gerlach/dsta-2025-1```

> ```cd dsta-2025-1```

2. Run the script

> ```python mnist_convnet.py```
