import os

from src.modules.dataset import Dataset
from src.modules.plotter import Plotter
from src.modules.polynomial_regression import PolynomialRegression

dataset = Dataset(
    os.path.join(os.getcwd(), '../datasets/regression_part_2.csv'),
    ['x'],
    'y',
    shuffle=True
)

dataset.normalize()

training_inputs, testing_inputs, training_outputs, testing_outputs = dataset.split_data(testing_size=0.3)

polynomial_regression = PolynomialRegression(
    training_inputs.copy(),
    training_outputs.copy(),
    testing_inputs.copy(),
    testing_outputs.copy(),
    k_order=2
)

plotter = Plotter(
    polynomial_regression,
    training_inputs.copy(),
    training_outputs.copy(),
    testing_inputs.copy(),
    testing_outputs.copy(),
    saves_output_dir=os.path.join(os.getcwd(), '../plots'),
    saves_prefix='polynomial'
)

polynomial_regression.train(max_iterations=30, use_armijo=True, learning_rate=0.01)

plotter.scatter_data(dataset.min(column='x'), dataset.max(column='x'), save=True)
plotter.plot_loss_evolution(save=True)