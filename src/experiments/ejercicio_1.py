import numpy as np

from src.data_management.preprocessing import normalize, normalize_with_params
from src.data_management.loader import load_csv
from src.network.multilayer_perceptron import MultilayerPerceptron
from src.network.neuron_layer import NeuronLayer
from src.activation.identity import IdentityActivation
from src.activation.tanh import TanhActivation
from src.activation.step import StepActivation
from src.cost.mse import MSECost
from src.optimizer.gradient_descent import GradientDescent
from src.trainer import Trainer
from src.config import ExperimentConfig
from analysis.plots import plot_regression
from src.metric.classify_data import classify_data
from src.metric.f1 import  F1Metric

def run(cfg: ExperimentConfig) -> None:
    """Ejercicio 1 — Detección de fraude.

    Entrena perceptrón simple lineal vs no lineal. Compara generalización.
    """

    target_columns = cfg.target_column
    excluded_columns = ["flagged_fraud"]

    dataset = load_csv(cfg.data_path, target_column=target_columns,columns_to_ignore=excluded_columns)

    if cfg.activation == "identity":
        activation = IdentityActivation()
    elif cfg.activation == "tanh":
        activation = TanhActivation(beta=cfg.beta)
    elif cfg.activation == "step":
        activation = StepActivation()
    else:
        raise ValueError(f"Activation '{cfg.activation}' no soportada en Ejercicio 1.")

    n_inputs = dataset.X.shape[1]
    model = MultilayerPerceptron([
        NeuronLayer(n_inputs=n_inputs, n_neurons=1, activation=activation),
    ])

    trainer = Trainer(
        cost_fn=MSECost(),
        optimizer=GradientDescent(learning_rate=cfg.eta),
        metrics=[],
        cfg=cfg,
    )

    Xmin = dataset.X.min(axis=0)
    Xmax = dataset.X.max(axis=0)
    Xmin_zeta = dataset.zeta.min(axis=0)
    Xmax_zeta = dataset.zeta.max(axis=0)

    norm_dataset = dataset.copy()
    norm_dataset.X = normalize_with_params(dataset.X, Xmin, Xmax)
    norm_dataset.zeta = normalize_with_params(dataset.zeta, Xmin_zeta, Xmax_zeta)

    history = trainer.fit(
        model,
        X_train=norm_dataset.X, zeta_train=norm_dataset.zeta,
        X_val=None, zeta_val=None,
    )

    layer = model.layers[0]
    print("=== ADALINE Lineal — verificación de arquitectura ===")
    #TODO: fix this. son varios pesos, y no layer.weights[0, 0].
    print(f"Peso final:  w={layer.weights[0, 0]:.4f}   (esperado ≈ 2.0)")
    print(f"Bias final:  w₀={layer.bias[0]:.4f}   (esperado ≈ 5.0)")
    print(f"Épocas:      {history['epochs']}")

    if dataset.X.shape[1] == 1:
        plot_regression(
            dataset.X, dataset.zeta, model,
            title="ADALINE Lineal — predicción vs datos",
            output_path="output/experiment/linear/regression.png",
            xlim=(-6, 6),
            ylim=(-8, 18),
        )
    else:
        print("Se omite plot_regression: solo aplica a datasets de 1 feature.")
    # TODO> para esta etapa, no usamos val. Como esta funcion de grafico depende de val, la comento por ahora
    # plot_error_curve(history, output_path="output/experiment/linear/error_curve.png")
    print("Gráficos guardados en output/experiment/linear/")

    print(f"Error final: {history['train_error'][-1]:.4f}")

    #TODO: pensar en otras maneras de splitting como k-fold.
    train_dataset, val_dataset, test_dataset,[train_index, val_index, test_index] = norm_dataset.split(
        train=cfg.split_train,
        val=cfg.split_val,
        test=cfg.split_test,
        seed=cfg.seed,
    )

    #TODO: entrenar
    #TODO: evaluar


    #TODO: aca donde clasifico, debo usar el output del NUEVO MODELO.
    # estoy usando el viejo, porque todavia no esta implementado el nuevo.
    # test_new_model_output_dataset = NEW_MODEL.forward(test_dataset.X)
    test_new_model_output_dataset = model.forward(test_dataset.X)
    test_zeta_dataset = load_csv(cfg.data_path, target_column=excluded_columns[0]).zeta[test_index]

    # TODO: add this as parameter in config. We assume that prob. >= 0.8 is a positive classification
    [false_pos, false_neg, true_pos, true_neg] = classify_data(test_zeta_dataset, test_new_model_output_dataset,threshold=0.8)
    print (f"Resultados en test: FP={false_pos}  FN={false_neg}  TP={true_pos}  TN={true_neg}")

    f1_metric = F1Metric().compute(false_pos, false_neg, true_pos, true_neg)
    print(f"Resultados en test: F1={f1_metric}")

