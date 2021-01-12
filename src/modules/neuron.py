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
        self.weights = []

        self.loss_history = []
        self.testing_loss_history = []

        self._initialize_weights()

    def _initialize_weights(self):
        input_dimension = self.inputs.shape[1]

        self.weights = np.array(
            [np.random.uniform(-1, 1) for _ in range(input_dimension)]
        )

    def _predict(self, x):
        return self.activation_function(np.dot(self.weights, x))

    def calculate_loss(self, testing=False):
        inputs = self.inputs
        outputs = self.outputs

        if testing:
            inputs = self.testing_inputs
            outputs = self.testing_outputs

        return self.loss_function(inputs, outputs, self._predict)

    def train(self):
        self.loss_history.append(self.calculate_loss())
        self.testing_loss_history.append(self.calculate_loss(testing=True))
