import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sign(x):
    return 1 if x > 0 else -1
