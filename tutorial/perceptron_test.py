import numpy as np
import matplotlib.pyplot as plt


from perceptron import Perceptron
from plot_utils import plot_decision_boundary


def main():
    """
       to visualize the problem, we are going to create a simple step perceptron
       to calculate the AND logic gate
       """

    # the AND logic gate has two inputs, these are all posible combinations
    # and we will use them as our training set
    X_and = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    # this is the expected output of our AND logic gate
    y_and = np.array([0, 0, 0, 1])

    p_and = Perceptron(learning_rate=0.03, epochs=2)
    p_and.fit(X_and, y_and)

    print("Weights:", p_and.weights)
    print("Bias:", p_and.bias)
    print("Predictions:", p_and.predict(X_and))
    plot_decision_boundary(X_and, y_and, p_and, "Perceptron Decision Boundary (AND)", "and_decision_boundary.png")



if __name__ == '__main__':
    main()


