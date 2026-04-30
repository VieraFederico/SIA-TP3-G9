import numpy as np

from tutorial.perceptron import Perceptron
from tutorial.plot_utils import plot_decision_boundary, plot_adaline_regression


def _tanh_derivative(x, beta_value=0.1):
    return beta_value * (1 - np.tanh(beta_value * x) ** 2)


class PerceptronNonLinear(Perceptron):

    #for this example, we will use the TANH(B * X) function as an activation function
    # where B is an arbitrary constant from > 0 to 10
    # the derivative from

    def __init__(self, learning_rate=0.1, epochs=20, epsilon=0.01):
        super().__init__(learning_rate, epochs,epsilon)
        self.beta_value = None

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        linear_output = np.dot(X, self.weights) + self.bias
        beta = 0.1 if self.beta_value is None else self.beta_value
        return np.tanh(beta * linear_output)

    def fit(self, X, y, beta_value=0.1):
        self.beta_value = beta_value
        n_samples, n_features = X.shape
        self.weights = np.random.rand(n_features)
        self.bias = np.random.rand()

        for i in range(self.epochs):
            print(f"Epoch: {i}")
            print("Weights:", self.weights)
            print("Bias:", self.bias)
            print("")

            predictions = np.zeros_like(y, dtype=float)
            rng = np.random.default_rng(seed=42)
            indices = rng.permutation(n_samples)

            for idx in indices:
                xi = X[idx]
                target = y[idx]

                linear_output = np.dot(xi, self.weights) + self.bias
                y_pred = np.tanh(self.beta_value * linear_output)
                predictions[idx] = y_pred

                update = self.lr * (target - y_pred)
                self.weights += update * xi * _tanh_derivative(linear_output, self.beta_value)
                self.bias += update * _tanh_derivative(linear_output, self.beta_value)

            err = mse(self,y, predictions)
            self.errors_per_epoch.append(err)
            print(f"Best Error {err}")
            print("")

            if err < self.epsilon:
                print(f"Method converged at epoch: {i}")
                break

            # inside fit(), after each epoch
            # plot_adaline_regression(
            #     X, y, self,
            #     f"TANH Regression Epoch {i + 1}",
            #     f"output/adaline_epoch_{i + 1}.png",
            #     xlim=(-6, 6),
            #     ylim=(-1.5,1.5),
            #     centered=True
            # )


def mse(self, zeta, predictions) -> float:
    """E = (1/2N) Σ (ζ - O)²"""
    N = len(zeta) if hasattr(zeta, '__len__') else 1
    return (1 / (2 * N)) * np.sum((zeta - predictions) ** 2)