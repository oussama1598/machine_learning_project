# Reference https://medium.com/analytics-vidhya/calculating-accuracy-of-an-ml-model-8ae7894802e
import numpy as np


def r_squared(inputs, outputs, predict_function, weights=None):
    SSE = sum([
        (outputs[i] - predict_function(x, weights)) ** 2
        for i, x in enumerate(inputs)
    ])

    SSW = sum([
        (outputs[i] - np.mean(outputs)) ** 2
        for i, x in enumerate(inputs)
    ])

    return 1 - (SSE / SSW)
