# Simple Convolutional Neural Network

## General Information

Simple convolutional neural network written from scratch in Python using NumPy library for vectorized computations. The network was trained and tested on _MNIST Handwritten Digit Classification Dataset_.

`mnist_digits_conv.py` - CNN model with max pooling and batch normalization

`mnist_digits.py` - Feed Forward Neural Network

File `layers.py` contains implementation of Linear, Convolutional, Pooling and BatchNorm layers which can be sequentially stacked together to create a Model instance from `model.py`. Additionally, `layers.py` implements ReLU and SoftMax activation functions.

Files `loss.py`, `optimizer.py` contain implementations of mean squared error and cross entropy loss functions and Adam optimizer.

## Technologies Used

Python, NumPy, Pandas (for one-hot encoding)
