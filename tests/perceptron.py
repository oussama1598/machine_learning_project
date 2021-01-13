import os
from src.modules.dataset import Dataset
from src.modules.perceptron import Perceptron
from src.modules.plotter import Plotter

dataset = Dataset(
    os.path.join(os.getcwd(), '../datasets/classification_data.csv'),
    ['x_1', 'x_2'],
    'y',
    shuffle=True
)

dataset.replace_output(lambda i, value: -1 if value == 0 else 1)

training_inputs, testing_inputs, training_outputs, testing_outputs = dataset.split_data(testing_size=0.3)

perceptron = Perceptron(
    training_inputs.copy(),
    training_outputs.copy(),
    testing_inputs.copy(),
    testing_outputs.copy()
)

plotter = Plotter(
    perceptron,
    training_inputs.copy(),
    training_outputs.copy(),
    testing_inputs.copy(),
    testing_outputs.copy(),
    saves_output_dir='../plots',
    saves_prefix='perceptron'
)

perceptron.train()

plotter.scatter_data_for_classification(dataset.min(column='x_1'), dataset.max(column='x_1'), save=True)
plotter.plot_loss_evolution(save=True)
