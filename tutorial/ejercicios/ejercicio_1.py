from analysis.plots import plot_error_curve
from config import ExperimentConfig
from data_management.loader import load_csv
from data_management.preprocessing import standardize, normalize
from tutorial.perceptron_linear import PerceptronLinear


def run(cfg: ExperimentConfig) -> None:
    target_columns = cfg.target_column
    excluded_columns = cfg.columns_to_ignore

    dataset = load_csv(cfg.data_path, target_column=target_columns, columns_to_ignore=excluded_columns)

    train_dataset, val_dataset, test_dataset = dataset.split(
        train=cfg.split_train,
        val=cfg.split_val,
        test=cfg.split_test,
        seed=cfg.seed,
    )

    norm_dataset=normalize(dataset.X)


    p_linear = PerceptronLinear(0.1, cfg.epochs, 0.01)
    p_linear.fit(norm_dataset,dataset.zeta)
    history = {"train_error": p_linear.errors_per_epoch, "val_error": p_linear.errors_per_epoch}
    plot_error_curve(history, "output/error_curve.png")
