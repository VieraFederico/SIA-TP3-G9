import numpy as np
from utils.plot_utils import plot_decision_boundary
from tutorial.perceptron import Perceptron


class PerceptronStep(Perceptron):


    #receives X as an input array. in this case a 2 integer array (because we are doing an AND logic gate)
    #np.dot calculates the dot product of X and the weights array. that is x1 * w1 + x2 * w2
    #np.where is similar to a Java predicate. if linear output >= 0, then returns 1, else returns 0
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)

    #receives X as an input array, and Y as the results array.
    #we run the fit function to train the perceptron and adjust weights and bias
    #ideally so that it can correctly solve the problem (in this case, logic AND problem)
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
            rng = np.random.default_rng(seed=42)
            indices = rng.permutation(n_samples)
            for idx in indices:
                xi = X[idx]
                target = y[idx]

                linear_output = np.dot(xi, self.weights) + self.bias
                y_pred = 1 if linear_output >= 0 else 0
                update = self.lr * (target - y_pred)
                self.weights += update * xi
                self.bias += update
                errors += int(target != y_pred)
            # if errors < self.epsilon:
            #     print("No errors, stopping training.")
            #     break
            self.errors_per_epoch.append(errors)
            plot_decision_boundary(X, y, self, f"Perceptron Decision Boundary (AND) Epoch {i+1}", f"output/epoch_{i+1}_decision_boundary.png")

