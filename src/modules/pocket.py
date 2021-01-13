import numpy as np

from src.helpers.loss_functions import normal_error
from src.helpers.math_functions import sign
from src.modules.neuron import Neuron


class Pocket(Neuron):
    def __init__(self, *args):
        super().__init__(
            *args,
            loss_function=normal_error,
            activation_function=sign
        )

    def predict(self, x):
        if len(x) != 1:
            return self._predict(x)

        return self._predict(np.concatenate((x, [1])))

    def predict_with_weights(self, x, weights):
        return self.activation_function(np.dot(weights, x))

    def calculate_loss_with_weights(self, weights):
        return self.loss_function(self.inputs, self.outputs, lambda x: self.predict_with_weights(x, weights))

    def train(self, max_iterations=1000):
        weights = self.weights.copy()

        for j in range(max_iterations):
            for i in range(len(self.inputs)):
                x = self.inputs[i]
                y = self.outputs[i]

                if self.predict_with_weights(x, weights) != y:
                    weights = weights + (y * x)

            if self.calculate_loss_with_weights(weights) < self.calculate_loss_with_weights(self.weights):
                self.weights = weights.copy()

            super().train()
