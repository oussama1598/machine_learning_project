import numpy as np

from src.helpers.loss_functions import mean_squared_error, mean_squared_gradient
from src.modules.neuron import Neuron


class PolynomialRegression(Neuron):
    def __init__(self, inputs, outputs, testing_inputs, testing_outputs, k_order=2):
        self.k_order = k_order

        for i in range(2, k_order + 1):
            inputs = np.array([np.concatenate((x, [x[0] ** i])) for x in inputs])
            testing_inputs = np.array([np.concatenate((x, [x[0] ** i])) for x in testing_inputs])

        super().__init__(inputs, outputs, testing_inputs, testing_outputs, loss_function=mean_squared_error)

    def predict(self, x):
        if len(x) != 1:
            return self._predict(x)

        for i in range(2, self.k_order + 1):
            x = np.concatenate((x, [x[0] ** i]))

        return self._predict(np.concatenate((x, [1])))

    def train(self, max_iterations=100, learning_rate=0.1):
        for i in range(max_iterations):
            gradients = mean_squared_gradient(self.inputs, self.outputs, self.predict)

            self.weights = self.weights - learning_rate * gradients

            super().train()
