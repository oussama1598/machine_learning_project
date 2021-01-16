import os

import numpy as np
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, model, training_inputs, training_outputs, testing_inputs,
                 testing_outputs, saves_output_dir='', saves_prefix=''):
        self.model = model

        self.training_inputs = training_inputs
        self.testing_inputs = testing_inputs

        self.training_outputs = training_outputs
        self.testing_outputs = testing_outputs

        self.saves_output_dir = saves_output_dir
        self.saves_prefix = saves_prefix

    def scatter_data(self, data_min, data_max, save=False, no_model=False):
        x_ = np.arange(data_min, data_max, 0.01)

        if not no_model:
            plt.plot(x_, [self.model.predict([x]) for x in x_], 'r', label='Model')

        plt.scatter(self.training_inputs, self.training_outputs, label='Training Data')
        plt.scatter(self.testing_inputs, self.testing_outputs, label='Testing Data')
        plt.legend(loc='upper left')

        if save:
            plt.savefig(
                os.path.join(self.saves_output_dir, f'{self.saves_prefix}_data.png')
            )

        plt.show()

    def scatter_data_for_classification(self, data_min, data_max, save=False):
        x_ = np.arange(data_min, data_max, 0.01)

        def hypotheses(x):
            return (-(self.model.weights[0] / self.model.weights[1]) * x) - (
                    self.model.weights[2] / self.model.weights[1])

        plt.plot(x_, [hypotheses(x) for x in x_], 'r', label='Model')

        plt.scatter(self.training_inputs[:, 0], self.training_inputs[:, 1], c=self.training_outputs,
                    label='Training Data')
        plt.scatter(self.testing_inputs[:, 0], self.testing_inputs[:, 1], label='Testing Data')
        plt.legend(loc='upper left')

        if save:
            plt.savefig(
                os.path.join(self.saves_output_dir, f'{self.saves_prefix}_data.png')
            )

        plt.show()

    def plot_loss_evolution(self, save=False):
        plt.plot(range(len(self.model.loss_history)), self.model.loss_history, label='Training Loss')
        plt.plot(range(len(self.model.testing_loss_history)), self.model.testing_loss_history, 'y',
                 label='Testing Loss')

        plt.legend(loc='upper left')

        if save:
            plt.savefig(
                os.path.join(self.saves_output_dir, f'{self.saves_prefix}_loss.png')
            )

        plt.show()

    def plot_accuracy_evolution(self, save=False):
        plt.plot(range(len(self.model.accuracy_history)), self.model.accuracy_history, label='Training Accuracy')
        plt.plot(range(len(self.model.testing_accuracy_history)), self.model.testing_accuracy_history, 'y',
                 label='Testing Accuracy')

        plt.legend(loc='upper left')

        if save:
            plt.savefig(
                os.path.join(self.saves_output_dir, f'{self.saves_prefix}_accuracy.png')
            )

        plt.show()
