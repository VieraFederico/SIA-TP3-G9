import numpy as np




from plot_utils import plot_decision_boundary, plot_adaline_regression
from tutorial.perceptron_linear import PerceptronLinear
from tutorial.perceptron_non_linear import PerceptronNonLinear
from tutorial.perceptron_step import PerceptronStep


def main():
    #perceptron_step_test()
    #perceptron_linear_test()
    tanh_perceptron_test()


def tanh_perceptron_test():
    X = np.linspace(-5, 5, 50)
    Y = np.tanh(X)
    perceptron = PerceptronNonLinear(learning_rate=0.15, epochs=100,epsilon=0.1)
    perceptron.fit(X.reshape(-1, 1), Y)
    print("Weights:", perceptron.weights)
    print("Bias:", perceptron.bias)
    print("Predictions:", perceptron.predict(X.reshape(-1, 1)))
    plot_adaline_regression(
        X, Y, perceptron,
        f"TANH Regression Epoch Result",
        f"tanh_result.png",
        xlim=(-6, 6),
        ylim=(-1.5, 1.5),
        centered=True
    )

def perceptron_linear_test():

    # 50 points in [-5, 5]
    X_linear = np.linspace(-5, 5, 50)
    Y_linear = 2 * X_linear + 5

    p_linear = PerceptronLinear(learning_rate=0.01, epochs=20,epsilon=0.1)
    p_linear.fit(X_linear.reshape(-1, 1), Y_linear)
    print("Weights:", p_linear.weights)
    print("Bias:", p_linear.bias)
    print("Predictions:", p_linear.predict(X_linear.reshape(-1, 1)))
    plot_adaline_regression(
        X_linear, Y_linear, p_linear,
        f"ADALINE Regression Epoch Result",
        f"adaline result.png",
        xlim=(-6, 6),
        ylim=(-8, 18),
        centered=True
    )

def perceptron_step_test():
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

    p_and = PerceptronStep(learning_rate=0.03, epochs=13,epsilon=0.1)
    p_and.fit(X_and, y_and)

    print("Weights:", p_and.weights)
    print("Bias:", p_and.bias)
    print("Predictions:", p_and.predict(X_and))
    plot_decision_boundary(X_and, y_and, p_and, "Perceptron Decision Boundary (AND)", "and_decision_boundary.png")


if __name__ == '__main__':
    main()


