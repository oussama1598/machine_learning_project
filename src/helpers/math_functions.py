import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sign(x):
    return 1 if x > 0 else -1


def leaky_relu(x):
    if x > 0:
        return x

    return 0
