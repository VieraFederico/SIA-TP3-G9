import numpy as np
from tutorial.plot_utils import plot_adaline_regression

from tutorial.perceptron import Perceptron


class PerceptronLinear(Perceptron):

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.dot(X, self.weights) + self.bias


    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.rand(n_features)
        self.bias = np.random.rand()
        for i in range(self.epochs):
            print(f"Epoch: {i}")
            print("Weights:", self.weights)
            print("Bias:", self.bias)
            print("")
            errors = 0
            for xi, target in zip(X, y):
                linear_output = np.dot(xi, self.weights) + self.bias
                # we instead assign y_pred as the identity function
                # (directly equals weighted sum instead of {0,1})
                y_pred = linear_output
                update = self.lr * (target - y_pred)

                # we need to update the weight and bias.
                # in the ADALINE algorithm, the activation function is the identity
                # so it does not influece the update rule in this case
                # in other cases, it should be appended as
                # self.weights += update * xi * (the derivative of activation function)
                self.weights += update * xi
                self.bias += update

                err = error(target, y_pred)
                errors += err
            self.errors_per_epoch.append(errors)
            if errors < self.epsilon:
                print(f"Method converged at epoch: {i}")
                break
            # inside fit(), after each epoch
            plot_adaline_regression(
                X, y, self,
                f"ADALINE Regression Epoch {i + 1}",
                f"output/adaline_epoch_{i + 1}.png",
                xlim=(-6, 6),
                ylim=(-8, 18),
                centered=True
            )


def error(actual, predicted):
        return 0.5 * np.sum(np.square(actual-predicted))