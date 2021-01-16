import numpy as np

from src.helpers.loss_functions import cross_entropy_error, cross_entropy_gradient
from src.helpers.math_functions import sigmoid
from src.modules.neuron import Neuron


class LogisticRegression(Neuron):
    def __init__(self, *args, activation_function=sigmoid, loss_function=cross_entropy_error):
        super().__init__(*args, activation_function=activation_function, loss_function=loss_function)

    def predict(self, x):
        if len(x) != 1:
            return self._predict(x)

        return self._predict(np.concatenate((x, [1])))

    def train(self, max_iterations=100, learning_rate=0.1, use_armijo=False):
        for i in range(max_iterations):
            gradients = cross_entropy_gradient(self.inputs, self.outputs, self.predict)

            if use_armijo:
                learning_rate = self.armijo()

            self.weights = self.weights - learning_rate * gradients

            super().train()
