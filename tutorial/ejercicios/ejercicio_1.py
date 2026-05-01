from analysis.plots import plot_error_curve
from src.config import ExperimentConfig
from src.data_management.loader import load_csv
from src.data_management.preprocessing import standardize, normalize
from tutorial.perceptron_linear import PerceptronLinear
from tutorial.perceptron_non_linear import PerceptronNonLinear


def run(cfg: ExperimentConfig) -> None:
    target_columns = cfg.target_column
    excluded_columns = cfg.columns_to_ignore

    dataset = load_csv(cfg.data_path, target_column=target_columns, columns_to_ignore=excluded_columns)

    norm_dataset = dataset.copy()
    norm_dataset.X=normalize(dataset.X)
    norm_dataset.zeta = normalize(dataset.zeta)

    p_linear = PerceptronLinear(0.1, cfg.epochs, 0.02)
    p_linear.fit(norm_dataset.X,norm_dataset.zeta)
    history = {"train_error": p_linear.errors_per_epoch, "val_error": p_linear.errors_per_epoch}
    plot_error_curve(history, "output/error_linear_curve.png")

    p_non_linear = PerceptronNonLinear(0.1, cfg.epochs, 0.02)
    p_non_linear.fit(norm_dataset.X,norm_dataset.zeta)

    history = {"train_error": p_linear.errors_per_epoch, "val_error": p_linear.errors_per_epoch}
    plot_error_curve(history, "output/error_linear_curve.png")

    #TODO: determinar con que tipo de metodologia seguir.

    train_dataset, val_dataset, test_dataset = norm_dataset.split(
        train=cfg.split_train,
        val=cfg.split_val,
        test=cfg.split_test,
        seed=cfg.seed,
    )