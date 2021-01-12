import numpy as np

from src.helpers.loss_functions import logit_error, logit_gradient
from src.helpers.math_functions import sigmoid
from src.modules.neuron import Neuron


class LogisticRegression(Neuron):
    def __init__(self, *args):
        super().__init__(*args, loss_function=logit_error)

    def predict(self, x):
        if len(x) != 1:
            return self._predict(x)

        return self._predict(np.concatenate((x, [1])))

    def train(self, max_iterations=100, learning_rate=0.1):
        for i in range(max_iterations):
            gradients = logit_gradient(self.inputs, self.outputs, self.predict)

            self.weights = self.weights - learning_rate * gradients

            super().train()
