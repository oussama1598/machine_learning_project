import csv
import math

import numpy as np
from tabulate import tabulate


class Dataset:
    def __init__(self, filepath, inputs_labels, output_label, shuffle=True):
        self.filepath = filepath
        self.input_labels = inputs_labels
        self.output_label = output_label
        self.shuffle = shuffle

        self.inputs = np.array([])
        self.outputs = np.array([])

        self._read_csv()

    def _read_csv(self):
        data = []

        with open(self.filepath, 'r') as file:
            reader = csv.DictReader(file)

            for row in reader:
                input_data = [float(row[label]) for label in self.input_labels]

                data.append(input_data + [float(row[self.output_label])])

        if self.shuffle:
            np.random.shuffle(data)

        self.inputs = np.array(data)[:, :-1]
        self.outputs = np.array(data)[:, -1]

    def replace_output(self, fn):
        for i in range(len(self.outputs)):
            self.outputs[i] = fn(i, self.outputs[i])

    def mean(self, column=''):
        if column in self.input_labels:
            feature_index = self.input_labels.index(column)

            return np.mean(self.inputs[:, feature_index])

        if column == self.output_label:
            return np.mean(self.outputs)

    def median(self, column=''):
        if column in self.input_labels:
            feature_index = self.input_labels.index(column)

            return np.median(self.inputs[:, feature_index])

        if column == self.output_label:
            return np.median(self.outputs)

    def max(self, column=''):
        if column in self.input_labels:
            feature_index = self.input_labels.index(column)

            return np.max(self.inputs[:, feature_index])

        if column == self.output_label:
            return np.max(self.outputs)

    def min(self, column=''):
        if column in self.input_labels:
            feature_index = self.input_labels.index(column)

            return np.min(self.inputs[:, feature_index])

        if column == self.output_label:
            return np.min(self.outputs)

    def normalize(self, ignore_output=False):
        if not ignore_output:
            self.outputs = (self.outputs - np.mean(self.outputs)) / np.std(self.outputs)

        means = []
        stds = []

        for i in range(self.inputs.shape[1]):
            means.append(np.mean(self.inputs[:, i]))
            stds.append(np.std(self.inputs[:, i]))

        for x in self.inputs:
            for i in range(len(x)):
                x[i] = (x[i] - means[i]) / stds[i]

    def split_data(self, testing_size=0.2):
        total_elements_size = self.inputs.shape[0]

        testing_size = math.ceil(total_elements_size * testing_size)

        training_inputs = self.inputs[:total_elements_size - testing_size]
        training_outputs = self.outputs[:total_elements_size - testing_size]

        testing_inputs = self.inputs[total_elements_size - testing_size:]
        testing_outputs = self.outputs[total_elements_size - testing_size:]

        return training_inputs, testing_inputs, training_outputs, testing_outputs

    def describe(self, columns=None):
        if columns is None:
            columns = []

        data = []

        for column in columns:
            data.append([
                column,
                self.max(column=column),
                self.min(column=column),
                self.mean(column=column),
                self.median(column=column)
            ])

        return tabulate(data, headers=['Name', 'Max', 'Min', 'Mean', 'Median'])

    def check_null_inputs(self):
        for x in self.inputs:
            for feature in x:
                if feature:
                    continue

                return False

        return True
