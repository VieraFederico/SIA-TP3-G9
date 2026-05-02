from src.metric.classify_data_mlp import classify_data_mlp
from src.data_management.dataset import Dataset
from src.metric.classify_data import classify_data
from src.metric.f1 import F1Metric
from src.cost.mse import MSECost
from src.optimizer.gradient_descent import GradientDescent
from src.activation.tanh import TanhActivation
from src.network.multilayer_perceptron import MultilayerPerceptron
from src.network.neuron_layer import NeuronLayer
from src.config import ExperimentConfig
import numpy as np
import ast
import pandas as pd

from src.trainer import Trainer


def run(cfg: ExperimentConfig) -> None:

    #SET UP
    df = pd.read_csv(cfg.data_path)
    X = np.array(df["image"].apply(ast.literal_eval).tolist())
    zeta = one_hot(df["label"].values)
    dataset = Dataset(X=X, zeta=zeta)

    model = MultilayerPerceptron([
        NeuronLayer(n_inputs=cfg.architecture[0], n_neurons=cfg.architecture[1], activation=TanhActivation(beta=1.0)),
        NeuronLayer(n_inputs=cfg.architecture[1], n_neurons=cfg.architecture[2], activation=TanhActivation(beta=1.0)),
    ])

    trainer_mlp = Trainer(
        cost_fn=MSECost(), optimizer=GradientDescent(learning_rate=cfg.eta), metrics=[], cfg=cfg,
    )

    train_dataset, val_dataset, test_dataset, [train_index, val_index, test_index] = dataset.split(
        train=cfg.split_train,
        val=cfg.split_val,
        test=cfg.split_test,
        seed=cfg.seed,
    )

    # LEARN
    # TODO EVALUATION

    history = trainer_mlp.fit(
        model, train_dataset.X, train_dataset.zeta, val_dataset.X, val_dataset.zeta
    )

    print(f"[DEBUG ej2] Training finished.")
    print(f"Error final: {history['train_error'][-1]:.4f}")


    # GENERALIZATION

    df = pd.read_csv("data/digits_test.csv")
    X = np.array(df["image"].apply(ast.literal_eval).tolist())
    zeta = df["label"].values
    dataset_2 = Dataset(X=X, zeta=zeta)

    test_output_dataset_2 = model.forward(dataset_2.X)
    confusion = classify_data_mlp(
        dataset_2.zeta, test_output_dataset_2)

    print("Confusion Matrix")
    print("Rows = True class")
    print("Cols = Predicted class\n")

    print("     " + " ".join(f"{i:5d}" for i in range(10)))

    for i, row in enumerate(confusion):
        print(f"{i:3d} | " + " ".join(f"{int(v):5d}" for v in row))


def one_hot(labels, n_classes=10):
    y = np.zeros((len(labels), n_classes))
    for i, label in enumerate(labels):
        y[i, label] = 1
    return y