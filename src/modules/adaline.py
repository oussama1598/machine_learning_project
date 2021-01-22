import numpy as np

from src.helpers.accuracy_functions import r_squared
from src.helpers.loss_functions import mean_squared_error
from src.helpers.math_functions import sign
from src.modules.neuron import Neuron


class Adaline(Neuron):
    def __init__(self, *args):
        super().__init__(
            *args,
            loss_function=mean_squared_error,
            activation_function=sign,
            accuracy_function=r_squared
        )

    def predict(self, x):
        if len(x) != 1:
            return self._predict(x)

        return self._predict(np.concatenate((x, [1])))

    def train(self, max_iterations=1000):
        for j in range(max_iterations):
            for i in range(len(self.inputs)):
                x = self.inputs[i]
                y = self.outputs[i]

                error = y - self.predict(x)

                if error != 0:
                    self.weights = self.weights + (2 * error * x)

            super().train()
