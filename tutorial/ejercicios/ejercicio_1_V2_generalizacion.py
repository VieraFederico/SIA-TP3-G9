import numpy as np

from data_management.dataset import Dataset
from data_management.loader import load_csv
from data_management.preprocessing import normalize
from data_management.splitter import k_fold_split_v2
from metric.classify_data import classify_data
from metric.f1 import F1Metric
from tutorial.perceptron import Perceptron
from tutorial.perceptron_non_linear import PerceptronNonLinear
from data_management.csv_utils import append_perceptron_result


def main():
    k = 10
    target_columns = "big_model_fraud_probability"
    excluded_columns = ["flagged_fraud", "timestamp", "device_screen_resolution"]

    dataset = load_csv("data/transactions.csv", target_column=target_columns, columns_to_ignore=excluded_columns)
    norm_dataset = normalize(dataset.X)

    folds = k_fold_split_v2(Dataset(norm_dataset,dataset.zeta), k=k, seed=42)
    # stores the best perceptron
    best_perceptron : Perceptron | None = None
    best_mean_f1_metric = 0.0

    beta_value = 0.3
    learning_rate = 0.1


    for beta_value in [0.1, 0.2, 0.3]:
        for learning_rate in [0.01, 0.05, 0.1]:
            best_perceptron_combo: Perceptron | None = None
            mean_f1_metric : float = 0.0
            f1_metrics = []
            for i, eval_set in enumerate(folds):
                # iterate through K folds, switch the eval_set in each iteration
                # join the remaining sets into one training set
                train_folds = [folds[j] for j in range(len(folds)) if j != i]
                X_train = np.concatenate([ds.X for ds in train_folds], axis=0)
                y_train = np.concatenate([ds.zeta for ds in train_folds], axis=0)
                y_eval_binary = (eval_set.zeta >= 0.85).astype(int)

                train_set = Dataset(X_train, y_train)
                perceptron = PerceptronNonLinear(learning_rate, 50, 1e-3)
                perceptron.fit(train_set.X,train_set.zeta,beta_value)

                #append to csv resulting weights, bias and perceptron.errors_per_epoch[-1]:.3f

                predictions = perceptron.predict(eval_set.X)


                [false_pos, false_neg, true_pos, true_neg] = classify_data(
                    y_eval_binary, predictions, threshold=0.85
                )
                print(f"Resultados en val (último fold): FP={false_pos}  FN={false_neg}  TP={true_pos}  TN={true_neg}")

                f1_metric = F1Metric().compute(false_pos, false_neg, true_pos, true_neg)
                perceptron.eval_metric_score = f1_metric

                f1_metrics.append(f1_metric)
                print(f"Resultados en val (último fold): F1={f1_metric}")


                if best_perceptron_combo is None or f1_metric > best_perceptron_combo.eval_metric_score:
                    best_perceptron_combo = perceptron


            mean_f1 = float(np.mean(f1_metrics))

            append_perceptron_result(
                output_path="output/perceptron_runs.csv",
                perceptron_type="non_linear",
                architecture=[7, 1],
                learning_rate=learning_rate,
                beta_value=beta_value,
                weights=best_perceptron_combo.weights,
                bias=best_perceptron_combo.bias,
                best_train_error=best_perceptron_combo.errors_per_epoch[-1],
                best_eval_error=mean_f1,
            )

            print(f"For Learning rate {learning_rate} and beta_value {beta_value}, Mean F1 = {mean_f1:.3f}")

            if best_perceptron is None or mean_f1 > best_mean_f1_metric:
                best_perceptron = best_perceptron_combo
                best_mean_f1_metric = mean_f1
                print(f"New best combo: lr={learning_rate}, beta={beta_value}, mean F1={mean_f1:.4f}")




    print("FINISHED")
    print(f"Best perceptron found with F1 mean score {best_mean_f1_metric}, learning rate {learning_rate} and beta_value {beta_value}")
    print(f"Weights: {best_perceptron.weights}")
    print(f"Bias: {best_perceptron.bias}")

        # Now use train_set and eval_set
        # e.g., model.fit(train_set.X, train_set.zeta)
        #       eval_pred = model.predict(eval_set.X)








if __name__ == '__main__':
    main()