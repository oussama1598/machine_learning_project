import numpy as np

from src.helpers.accuracy_functions import r_squared
from src.helpers.loss_functions import normal_error
from src.helpers.math_functions import sign
from src.modules.neuron import Neuron


class Pocket(Neuron):
    def __init__(self, *args):
        super().__init__(
            *args,
            loss_function=normal_error,
            activation_function=sign,
            accuracy_function=r_squared
        )

    def predict(self, x):
        if len(x) != 1:
            return self._predict(x)

        return self._predict(np.concatenate((x, [1])))

    def train(self, max_iterations=1000):
        weights = self.weights.copy()

        for j in range(max_iterations):
            for i in range(len(self.inputs)):
                x = self.inputs[i]
                y = self.outputs[i]

                if self._predict(x, weights) != y:
                    weights = weights + (y * x)

            if self.calculate_loss(weights) < self.calculate_loss(self.weights):
                self.weights = weights.copy()

            super().train()
