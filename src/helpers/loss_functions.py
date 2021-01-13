import math

import numpy as np


def mean_squared_error(inputs, outputs, predict_function):
    return sum([
        (outputs[i] - predict_function(x)) ** 2
        for i, x in enumerate(inputs)
    ]) / inputs.shape[0]


def mean_squared_gradient(inputs, outputs, predict_function):
    gradients = []

    for j in range(inputs.shape[1]):
        gradient = 0

        for i in range(inputs.shape[0]):
            x = inputs[i]

            gradient += x[j] * (predict_function(x) - outputs[i])

        gradient *= (2.0 / inputs.shape[0])
        gradients.append(gradient)

    return np.array(gradients)


def logit_error(inputs, outputs, predict_function):
    return - sum([
        (outputs[i] - math.log(predict_function(x))) + ((1 + outputs[i]) - math.log(1 - predict_function(x)))
        for i, x in enumerate(inputs)
    ]) / inputs.shape[0]


def logit_gradient(inputs, outputs, predict_function):
    gradients = []

    for j in range(inputs.shape[1]):
        gradient = 0

        for i in range(inputs.shape[0]):
            x = inputs[i]

            gradient += x[j] * (predict_function(x) - outputs[i])

        gradients.append(gradient)

    return np.array(gradients)


def normal_error(inputs, outputs, predict_function):
    return sum([
        int(predict_function(inputs[i]) != outputs[i])
        for i in range(len(inputs))
    ]) / inputs.shape[0]
