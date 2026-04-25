import numpy as np

from tutorial.perceptron import Perceptron
from tutorial.plot_utils import plot_decision_boundary


class PerceptronLinear(Perceptron):

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.dot(X, self.weights) + self.bias


    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
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
                err = error(target, y_pred)
                update = self.lr * (target - y_pred)

                # we need to update the weight and bias.
                # in the ADALINE algorithm, the activation function is the identity
                # so it does not influece the update rule in this case
                # in other cases, it should be appended as
                # self.weights += update * xi * (the derivative of activation function)
                self.weights += update * xi
                self.bias += update
                errors += err
            self.errors_per_epoch.append(errors)
            # import
            from tutorial.plot_utils import plot_adaline_regression

            # inside fit(), after each epoch
            plot_adaline_regression(
                X, y, self,
                f"ADALINE Regression Epoch {i + 1}",
                f"output/adaline_epoch_{i + 1}.png"
            )


def error(actual, predicted):
        return 0.5 * np.sum(np.square(actual-predicted))