import dataclasses
import numpy as np

from src.data_management.preprocessing import normalize, normalize_with_params
from src.data_management.loader import load_csv
from src.data_management.splitter import k_fold_split
from src.data_management.dataset import Dataset
from src.network.multilayer_perceptron import MultilayerPerceptron
from src.network.neuron_layer import NeuronLayer
from src.activation.identity import IdentityActivation
from src.activation.tanh import TanhActivation
from src.activation.step import StepActivation
from src.cost.mse import MSECost
from src.optimizer.gradient_descent import GradientDescent
from src.trainer import Trainer
from src.config import ExperimentConfig
from analysis.plots import plot_regression, plot_error_curve, plot_learning_comparison
from src.metric.classify_data import classify_data
from src.metric.f1 import F1Metric


def _learning_study(cfg: ExperimentConfig, dataset: Dataset) -> tuple[dict, dict]:
    """Ejercicio 1a/1b: Compara capacidad de aprendizaje lineal vs no lineal.

    Entrena sobre el dataset COMPLETO normalizado sin split — esto es un
    estudio de capacidad, no de generalización.
    """
    # Normalizar dataset completo — en un capacity study esto es aceptable
    Xmin = dataset.X.min(axis=0)
    Xmax = dataset.X.max(axis=0)
    Xmin_zeta = dataset.zeta.min(axis=0)
    Xmax_zeta = dataset.zeta.max(axis=0)
    norm_X = normalize_with_params(dataset.X, Xmin, Xmax)
    norm_zeta = normalize_with_params(dataset.zeta, Xmin_zeta, Xmax_zeta)

    n_inputs = dataset.X.shape[1]

    # Config fija para la fase de comparación de capacidad (independiente de cfg)
    EPOCHS = 200
    ETA = 0.01
    MODE = "online"
    study_cfg = dataclasses.replace(cfg, epochs=EPOCHS, training_mode=MODE)

    # --- Perceptrón lineal (ADALINE) ---
    # X_val=None es intencional — esta fase no usa validación
    model_linear = MultilayerPerceptron([
        NeuronLayer(n_inputs=n_inputs, n_neurons=1, activation=IdentityActivation()),
    ])
    trainer_linear = Trainer(
        cost_fn=MSECost(),
        optimizer=GradientDescent(learning_rate=ETA),
        metrics=[],
        cfg=study_cfg,
    )
    history_linear = trainer_linear.fit(
        model_linear, norm_X, norm_zeta, X_val=None, zeta_val=None
    )

    layer = model_linear.layers[0]
    print("=== ADALINE Lineal — verificación de arquitectura ===")
    #TODO: fix this. son varios pesos, y no layer.weights[0, 0].
    print(f"Peso final:  w={layer.weights[0, 0]:.4f}   (esperado ≈ 2.0)")
    print(f"Bias final:  w₀={layer.bias[0]:.4f}   (esperado ≈ 5.0)")
    print(f"Épocas:      {history_linear['epochs']}")
    # TODO> para esta etapa, no usamos val. Como esta funcion de grafico depende de val, la comento por ahora
    print(f"Error final (lineal): {history_linear['train_error'][-1]:.4f}")

    # --- Perceptrón no lineal (Tanh) ---
    # X_val=None es intencional — esta fase no usa validación
    model_nonlinear = MultilayerPerceptron([
        NeuronLayer(n_inputs=n_inputs, n_neurons=1, activation=TanhActivation(beta=cfg.beta)),
    ])
    trainer_nonlinear = Trainer(
        cost_fn=MSECost(),
        optimizer=GradientDescent(learning_rate=ETA),
        metrics=[],
        cfg=study_cfg,
    )
    history_nonlinear = trainer_nonlinear.fit(
        model_nonlinear, norm_X, norm_zeta, X_val=None, zeta_val=None
    )
    print(f"Error final (no lineal): {history_nonlinear['train_error'][-1]:.4f}")

    # TODO Ejercicio 1a/1b: analyze underfitting and saturation from these curves to select the best perceptron for Part 2
    plot_learning_comparison(
        history_linear, history_nonlinear,
        output_path="output/experiment/ej1/learning_comparison.png",
    )
    print("Gráficos guardados en output/experiment/ej1/")

    return history_linear, history_nonlinear


def _generalization_study(cfg: ExperimentConfig, dataset: Dataset, model: MultilayerPerceptron) -> None:
    """Ejercicio 1c: Estudia generalización usando k-fold cross-validation.

    Recibe dataset SIN normalizar — la normalización se computa por fold
    usando solo los datos de entrenamiento de ese fold.
    """
    #TODO: pensar en otras maneras de splitting como k-fold.
    # TODO: make k configurable via cfg (add k_folds field to ExperimentConfig)
    k = 5
    splits = k_fold_split(dataset, k=k, seed=cfg.seed)

    # Mirror the same permutation used inside k_fold_split to track original row indices.
    # TODO: extend k_fold_split to return indices and remove this coupling.
    n = len(dataset.X)
    fold_index_groups = np.array_split(np.random.default_rng(cfg.seed).permutation(n), k)
    fraud_labels_full = load_csv(cfg.data_path, target_column="flagged_fraud").zeta

    val_errors_final = []
    last_fold_model = None
    last_norm_val_X = None
    last_val_fraud_labels = None

    for i, (train_ds, val_ds) in enumerate(splits):
        # Compute normalization params ONLY from train_ds — never from val or test
        Xmin = train_ds.X.min(axis=0)
        Xmax = train_ds.X.max(axis=0)
        Xmin_zeta = train_ds.zeta.min(axis=0)
        Xmax_zeta = train_ds.zeta.max(axis=0)

        norm_train_X = normalize_with_params(train_ds.X, Xmin, Xmax)
        norm_train_zeta = normalize_with_params(train_ds.zeta, Xmin_zeta, Xmax_zeta)
        norm_val_X = normalize_with_params(val_ds.X, Xmin, Xmax)
        norm_val_zeta = normalize_with_params(val_ds.zeta, Xmin_zeta, Xmax_zeta)

        # Fresh model weights each fold to avoid weight contamination from previous folds
        fold_model = MultilayerPerceptron([
            NeuronLayer(n_inputs=layer.n_inputs, n_neurons=layer.n_neurons, activation=layer.activation)
            for layer in model.layers
        ])

        #TODO: entrenar
        trainer = Trainer(
            cost_fn=MSECost(),
            optimizer=GradientDescent(learning_rate=cfg.eta),
            metrics=[],
            cfg=cfg,
        )
        history = trainer.fit(
            fold_model,
            X_train=norm_train_X, zeta_train=norm_train_zeta,
            X_val=norm_val_X, zeta_val=norm_val_zeta,
        )

        #TODO: evaluar
        plot_error_curve(history, output_path=f"output/experiment/ej1/fold_{i}_error.png")
        if history["val_error"]:
            print(f"Fold {i}: train_error={history['train_error'][-1]:.4f}  val_error={history['val_error'][-1]:.4f}")
            val_errors_final.append(history["val_error"][-1])

        last_fold_model = fold_model
        last_norm_val_X = norm_val_X
        last_val_fraud_labels = fraud_labels_full[fold_index_groups[i]]

    avg_val_error = sum(val_errors_final) / len(val_errors_final) if val_errors_final else float("nan")
    print(f"\nPromedio val_error (k={k} folds): {avg_val_error:.4f}")

    # TODO Ejercicio 1c: select best model and recommend fraud detection threshold

    #TODO: aca donde clasifico, debo usar el output del NUEVO MODELO.
    # estoy usando el viejo, porque todavia no esta implementado el nuevo.
    # test_new_model_output_dataset = NEW_MODEL.forward(test_dataset.X)
    last_val_predictions = np.array([last_fold_model.forward(xi) for xi in last_norm_val_X])

    # TODO: add this as parameter in config. We assume that prob. >= 0.8 is a positive classification
    [false_pos, false_neg, true_pos, true_neg] = classify_data(
        last_val_fraud_labels, last_val_predictions, threshold=0.8
    )
    print(f"Resultados en val (último fold): FP={false_pos}  FN={false_neg}  TP={true_pos}  TN={true_neg}")

    f1_metric = F1Metric().compute(false_pos, false_neg, true_pos, true_neg)
    print(f"Resultados en val (último fold): F1={f1_metric}")


def run(cfg: ExperimentConfig) -> None:
    """Ejercicio 1 — Detección de fraude.

    Entrena perceptrón simple lineal vs no lineal. Compara generalización.
    """
    target_columns = cfg.target_column
    excluded_columns = ["flagged_fraud"]

    dataset = load_csv(cfg.data_path, target_column=target_columns, columns_to_ignore=excluded_columns)

    # Parte 1: comparar capacidad de aprendizaje lineal vs no lineal (Ejercicio 1a/1b)
    _learning_study(cfg, dataset)

    # Parte 2: estudio de generalización con el modelo no lineal (Tanh)
    # Tanh se prefiere sobre lineal: la salida acotada es más apropiada para
    # estimación de probabilidad de fraude que la salida lineal no acotada.
    # TODO: Verify this choice against Part 1 curves before final submission
    n_inputs = dataset.X.shape[1]
    model = MultilayerPerceptron([
        NeuronLayer(n_inputs=n_inputs, n_neurons=1, activation=TanhActivation(beta=cfg.beta)),
    ])
    _generalization_study(cfg, dataset, model)
