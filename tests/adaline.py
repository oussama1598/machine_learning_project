import os

from src.modules.adaline import Adaline
from src.modules.dataset import Dataset
from src.modules.plotter import Plotter


def noisify_data(_dataset, i, value, data_size):
    data_size = int(data_size * len(_dataset.outputs))

    if i <= data_size:
        return -1 if value == 1 else 1

    return value


dataset = Dataset(
    os.path.join(os.getcwd(), '../datasets/classification_data.csv'),
    ['x_1', 'x_2'],
    'y',
    shuffle=True
)

dataset.replace_output(lambda i, value: -1 if value == 0 else 1)
dataset.replace_output(lambda i, value: noisify_data(dataset, i, value, 0.07))

training_inputs, testing_inputs, training_outputs, testing_outputs = dataset.split_data(testing_size=0.3)

adaline = Adaline(
    training_inputs.copy(),
    training_outputs.copy(),
    testing_inputs.copy(),
    testing_outputs.copy()
)

plotter = Plotter(
    adaline,
    training_inputs.copy(),
    training_outputs.copy(),
    testing_inputs.copy(),
    testing_outputs.copy(),
    saves_output_dir='../plots',
    saves_prefix='adaline'
)

adaline.train()

plotter.scatter_data_for_classification(dataset.min(column='x_1'), dataset.max(column='x_1'), save=True)
plotter.plot_loss_evolution(save=True)
