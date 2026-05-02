from data_management.loader import load_csv
from data_management.preprocessing import normalize
from tutorial.perceptron_linear import PerceptronLinear
from tutorial.perceptron_non_linear import PerceptronNonLinear
from data_management.csv_utils import append_perceptron_result
from utils.plot_utils import plot_error_scatter


def main():
    target_columns = "big_model_fraud_probability"
    excluded_columns = ["flagged_fraud", "timestamp", "device_screen_resolution"]

    dataset = load_csv("data/transactions.csv", target_column=target_columns, columns_to_ignore=excluded_columns)
    X = normalize(dataset.X)
    y = dataset.zeta

    n_runs = 3
    results = {}  # label -> list of final errors

    # Linear combos
    for lr in [0.01, 0.05, 0.1]:
        label = f"linear-lr={lr:.2f}"
        results[label] = []
        for _ in range(n_runs):
            p = PerceptronLinear(lr, 75, 1e-3)
            p.fit(X, y)
            results[label].append(p.errors_per_epoch[-1])
            append_perceptron_result("output/perceptron_runs.csv", "linear",
                                     [X.shape[1], 1], lr, None, p.weights, p.bias, p.errors_per_epoch[-1], None)
    print("FINISHED LINEAR COMBO")

    # Non-linear combos
    for lr in [0.01, 0.05, 0.1]:
        for beta in [0.1, 0.2, 0.3]:
            label = f"tanh-lr={lr:.2f}-b={beta:.2f}"
            results[label] = []
            for _ in range(n_runs):
                p = PerceptronNonLinear(lr, 75, 1e-3)
                p.fit(X, y, beta)
                results[label].append(p.errors_per_epoch[-1])
                append_perceptron_result("output/perceptron_runs.csv", "non-linear", [X.shape[1], 1],lr,beta, p.weights, p.bias,
                                         p.errors_per_epoch[-1], None)

    plot_error_scatter(results, "output/adaline_boxplot.png")


if __name__ == '__main__':
    main()
