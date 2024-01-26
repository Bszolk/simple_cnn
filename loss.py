import numpy as np


def mean_squared_error(y_hat, y, derivative=False):
    if derivative:
        return 2 * (y_hat - y)
    return np.power((y_hat - y), 2)


def categorical_cross_entropy(y_hat, y, derivative=False):
    if derivative:
        return y_hat - y
    return -np.log(y_hat) * y


