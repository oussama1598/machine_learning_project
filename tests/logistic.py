import os

from src.modules.dataset import Dataset
from src.modules.logistic_regression import LogisticRegression
from src.modules.plotter import Plotter

dataset = Dataset(
    os.path.join(os.getcwd(), '../datasets/classification_data.csv'),
    ['x_1', 'x_2'],
    'y',
    shuffle=True
)

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
    testing_outputs.copy(),
    saves_output_dir='../plots',
    saves_prefix='logistic'
)

logistic_regression.train(max_iterations=1000, learning_rate=0.01)

plotter.scatter_data_for_classification(dataset.min(column='x_1'), dataset.max(column='x_1'), save=True)
plotter.plot_loss_evolution(save=True)
