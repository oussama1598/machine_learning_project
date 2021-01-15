import numpy as np

from src.helpers.loss_functions import mean_squared_error, mean_squared_gradient
from src.modules.neuron import Neuron


class LinearRegression(Neuron):
    def __init__(self, inputs, outputs, testing_inputs, testing_outputs):
        super().__init__(inputs, outputs, testing_inputs, testing_outputs, loss_function=mean_squared_error)

    def predict(self, x):
        if len(x) != 1:
            return self._predict(x)

        return self._predict(np.concatenate((x, [1])))

    def train(self, max_iterations=100, learning_rate=0.1, use_armijo=False):
        for i in range(max_iterations):
            gradients = mean_squared_gradient(self.inputs, self.outputs, self.predict)

            if use_armijo:
                learning_rate = self.armijo()

            self.weights = self.weights - learning_rate * gradients

            super().train()
