import numpy as np

from tutorial.perceptron import Perceptron
from tutorial.plot_utils import plot_decision_boundary, plot_adaline_regression


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
            predictions = np.zeros_like(y, dtype=float)
            rng = np.random.default_rng(seed=42)
            indices = rng.permutation(n_samples)
            for idx in indices:
                xi = X[idx]
                target = y[idx]

                linear_output = np.dot(xi, self.weights) + self.bias
                # we instead assign y_pred as the identity function
                # (directly equals weighted sum instead of {0,1})
                y_pred = linear_output
                predictions[idx] = y_pred


                update = self.lr * (target - y_pred)

                # we need to update the weight and bias.
                # in the ADALINE algorithm, the activation function is the identity
                # so it does not influece the update rule in this case
                # in other cases, it should be appended as
                # self.weights += update * xi * (the derivative of activation function)
                self.weights += update * xi
                self.bias += update


            err = mse(self,y, predictions)
            self.errors_per_epoch.append(err)
            print(f"Best Error {err}")
            print("")

            # import
            # inside fit(), after each epoch
            # plot_adaline_regression(
            #     X, y, self,
            #     f"ADALINE Regression Epoch {i + 1}",
            #     f"output/adaline_epoch_{i + 1}.png",
            #     xlim=(-6, 6),
            #     ylim=(-8, 18),
            #     centered=True
            # )


def mse(self, zeta, predictions) -> float:
    """E = (1/2N) Σ (ζ - O)²"""
    N = len(zeta) if hasattr(zeta, '__len__') else 1
    return (1 / (2 * N)) * np.sum((zeta - predictions) ** 2)