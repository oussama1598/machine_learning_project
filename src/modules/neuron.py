import numpy as np


class Neuron:
    def __init__(self, inputs: np.array, outputs: np.array, testing_inputs: np.array, testing_outputs: np.array,
                 loss_function, activation_function=lambda x: x):
        self.inputs = np.array([np.concatenate((x, np.array([1]))) for x in inputs])
        self.outputs = outputs

        self.testing_inputs = np.array([np.concatenate((x, np.array([1]))) for x in testing_inputs])
        self.testing_outputs = testing_outputs

        self.loss_function = loss_function
        self.activation_function = activation_function
        self.weights = np.array([])

        self.loss_history = []
        self.testing_loss_history = []

        self._initialize_weights()

    def _initialize_weights(self):
        input_dimension = self.inputs.shape[1]

        self.weights = np.array(
            [np.random.uniform(-1, 1) for _ in range(input_dimension)]
        )

    def _predict(self, x, weights=None):
        if weights is None:
            weights = self.weights

        return self.activation_function(np.dot(weights, x))

    def calculate_loss(self, weights=None, testing=False):
        inputs = self.inputs
        outputs = self.outputs

        if testing:
            inputs = self.testing_inputs
            outputs = self.testing_outputs

        return self.loss_function(inputs, outputs, self._predict, weights)

    def armijo_gradient(self):
        partial_derivatives = []

        for i in range(len(self.weights)):
            dx = np.zeros(len(self.weights))
            dx[i] = 0.001

            partial_derivatives.append(
                (self.loss_function(self.inputs, self.outputs, self._predict, self.weights + dx) - self.loss_function(
                    self.inputs, self.outputs, self._predict, self.weights)) / 0.001
            )

        return np.array(partial_derivatives)

    def armijo(self, beta=0.1):
        epsilon = 1
        derivative = self.armijo_gradient()

        def error(x):
            return self.loss_function(self.inputs, self.outputs, self._predict, x)

        while error(self.weights - epsilon * derivative) > error(self.weights) - (epsilon / 2) * (
                np.linalg.norm(derivative) ** 2):
            epsilon = epsilon * beta

        return epsilon

    def train(self):
        self.loss_history.append(self.calculate_loss())
        self.testing_loss_history.append(self.calculate_loss(testing=True))
