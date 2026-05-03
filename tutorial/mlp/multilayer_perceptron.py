from typing import List, Any

import numpy as np
from numpy import dtype, float64, ndarray

from activation.activation import Array
from tutorial.perceptron import Perceptron


class Neuron:
    def __init__(self, n_inputs, n_outputs, weights, bias, activation_function) -> None:
        self.inputs: List[float] = [0.0] * n_inputs
        self.outputs: List[float] = [0.0] * n_outputs
        self.weights: List[float] = [0.0] * n_outputs
        self.bias: float = bias
        self.activation_function = activation_function


    # "feed forward pass" each neuron computes its outputs and spits the output to the next layer
    def calculate_output(self, inputs: Array) -> float:
        linear_output = np.dot(inputs, self.weights) + self.bias
        prediction = self.activation_function(linear_output)
        return prediction



class MultilayerPerceptron:

    def predict(self, X: Array) -> Array:
        n_samples, _ = X.shape
        n_outputs = len(self.architecture[-1]) # last layer of neurons, each produces a different output
        # we will have n_sample rows of n_output columns, a column represents a last layer neuron
        outputs = np.zeros((n_samples, n_outputs), dtype=float)
        for idx in range(n_samples):
            outputs[idx] = self.forward_pass(X[idx]) #the row idx will contain all outputs of neurons
        return outputs

    def __init__(self, architecture: List[int], epochs=50) -> None:
        # Create perceptron layers
        # [7, 3 ,2] has 2 Neuron layers, 7 is the dataset number of inputs
        # and the last layer has every neuron with one input
        self.architecture: List[List[Neuron]] = []
        self.epochs = epochs
        for index in range(len(architecture)):
            if index!=0:
                n_inputs = architecture[index - 1]
                n_neurons = architecture[index]
                is_last = (index == len(architecture) - 1)
                n_outputs = 1 if is_last else architecture[index + 1]

                layer: List[Neuron] = []
                for _ in range(n_neurons):
                    layer.append(Neuron(n_inputs, n_outputs, np.zeros(n_outputs), np.zeros(n_outputs), identity_function))
                self.architecture.append(layer)

    def fit(self, X_train: Array, zeta_train: Array, X_val: Array | None, zeta_val: Array | None):
        for epoch in range(self.epochs):
            n_samples, _ = X_train.shape
            rng = np.random.default_rng(seed=42)
            indices = rng.permutation(n_samples)

            # store outputs for each sample (n_samples x n_outputs)
            # n_outputs = number of neurons in last layer
            n_outputs = len(self.architecture[-1])
            outputs = np.zeros((n_samples, n_outputs), dtype=float)

            for idx in indices:
                x = X_train[idx]  # row of the dataset containing al input columns
                y_pred = self.forward_pass(x)  # produces a row of outputs that has n_outputs columns
                outputs[idx] = y_pred  # append the row of outputs to the outputs array

                # outputs now holds model predictions for the epoch
                # TODO: compute error + backprop

    def forward_pass(self, layer_inputs: Array) -> Array:
        # layer_inputs is a 1D array for one sample
        for layer in self.architecture:
            # each neuron produces a scalar output
            layer_outputs = np.array([n.calculate_output(layer_inputs) for n in layer], dtype=float)
            layer_inputs = layer_outputs  # feed into next layer
        return layer_inputs  # final layer output (1D array)
        # TODO: compute error and backprop


def identity_function(x):
    return x

if __name__ == '__main__':
    mlp = MultilayerPerceptron([7, 3,2])
    print(f"len{len(mlp.architecture)}")
    for i, lay in enumerate(mlp.architecture):
        print(f"Layer {i} has {len(lay)} neurons")
        for j, neuron in enumerate(lay):
            print(f"  Neuron {j} has {len(neuron.inputs)} inputs and {len(neuron.outputs)} outputs")
