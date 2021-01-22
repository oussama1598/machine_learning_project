import numpy as np

from src.helpers.accuracy_functions import r_squared
from src.helpers.loss_functions import normal_error
from src.helpers.math_functions import sign
from src.modules.neuron import Neuron


class Perceptron(Neuron):
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

    def train(self):
        while self.calculate_loss() != 0:
            for i in range(len(self.inputs)):
                x = self.inputs[i]
                y = self.outputs[i]

                if self.predict(x) * y <= 0:
                    self.weights = self.weights + (x * y)

            super().train()
