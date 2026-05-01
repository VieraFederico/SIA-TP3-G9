import numpy as np
import matplotlib.pyplot as plt

from data_management.loader import load_csv
from data_management.preprocessing import normalize
from tutorial.perceptron_linear import PerceptronLinear
from tutorial.perceptron_non_linear import PerceptronNonLinear


def main():
    target_columns = "big_model_fraud_probability"
    excluded_columns = ["flagged_fraud", "timestamp", "device_screen_resolution"]

    dataset = load_csv("data/transactions.csv", target_column=target_columns, columns_to_ignore=excluded_columns)
    X = normalize(dataset.X)
    y = dataset.zeta

    n_runs = 10
    results = {}  # label -> list of final errors

    # Linear combos
    for lr in [0.01, 0.05, 0.1]:
        label = f"linear-lr={lr:.2f}"
        results[label] = []
        for _ in range(n_runs):
            p = PerceptronLinear(lr, 75, 1e-3)
            p.fit(X, y)
            results[label].append(p.errors_per_epoch[-1])

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

    plot_error_scatter(results, "output/adaline_boxplot.png")


def plot_error_scatter(results, output_path):
    labels = list(results.keys())

    plt.figure(figsize=(10, 5))
    for i, label in enumerate(labels):
        y = results[label]
        x = np.full(len(y), i, dtype=float)
        jitter = (np.random.rand(len(y)) - 0.5) * 0.2  # small horizontal jitter
        plt.scatter(x + jitter, y, alpha=0.7)

    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.title("Final Error Scatter by Hyperparameter Combo")
    plt.ylabel("Final MSE (last epoch)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == '__main__':
    main()