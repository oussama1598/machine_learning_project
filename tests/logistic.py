import os

from src.modules.dataset import Dataset
from src.modules.logistic_regression import LogisticRegression
from src.modules.plotter import Plotter

dataset = Dataset(
    os.path.join(os.getcwd(), '../datasets/regression_part_2.csv'),
    ['x'],
    'y',
    shuffle=True
)

dataset.normalize()

training_inputs, testing_inputs, training_outputs, testing_outputs = dataset.split_data(testing_size=0.3)

logistic_regression = LogisticRegression(
    training_inputs.copy(),
    training_outputs.copy(),
    testing_inputs.copy(),
    testing_outputs.copy()
)

plotter = Plotter(
    logistic_regression,
    training_inputs.copy(),
    training_outputs.copy(),
    testing_inputs.copy(),
    testing_outputs.copy()
)

logistic_regression.train(max_iterations=1000, learning_rate=0.1)

plotter.scatter_data(dataset.min(column='x'), dataset.max(column='x'))
plotter.plot_loss_evolution()
