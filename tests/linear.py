import os

from src.modules.dataset import Dataset
from src.modules.linear_regression import LinearRegression
from src.modules.plotter import Plotter

dataset = Dataset(
    os.path.join(os.getcwd(), '../datasets/regression_part_1.csv'),
    ['x'],
    'y',
    shuffle=True
)

dataset.normalize()

training_inputs, testing_inputs, training_outputs, testing_outputs = dataset.split_data(testing_size=0.3)

linear_regression = LinearRegression(
    training_inputs.copy(),
    training_outputs.copy(),
    testing_inputs.copy(),
    testing_outputs.copy()
)

plotter = Plotter(
    linear_regression,
    training_inputs.copy(),
    training_outputs.copy(),
    testing_inputs.copy(),
    testing_outputs.copy(),
    saves_output_dir=os.path.join(os.getcwd(), '../plots'),
    saves_prefix='linear'
)

linear_regression.train(max_iterations=100, learning_rate=0.01, use_armijo=True)

plotter.scatter_data(dataset.min(column='x'), dataset.max(column='x'), save=True)
plotter.plot_loss_evolution(save=True)
plotter.plot_accuracy_evolution(save=True)

training_loss, testing_loss = linear_regression.get_loss()
training_accuracy, testing_accuracy = linear_regression.get_accuracy()

print(f"""Training Accuracy: {training_accuracy}, Training Loss: {training_loss}
Testing Accuracy: {testing_accuracy}, Testing Loss: {testing_loss}
""")
