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
    zeta = df["label"].values
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
    history = trainer_mlp.fit(
        model, train_dataset.X, train_dataset.zeta, X_val=None, zeta_val=None
    )

    # DEBUG: training history
    print(f"[DEBUG ej2] Training finished.")


    #TODO
    # EVALUATION

    # GENERALIZATION
    test_output_dataset = model.forward(test_dataset.X)
    confusion = classify_data_mlp(
        test_dataset.zeta, test_output_dataset
    )

    print("Confusion Matrix")
    print("Rows = True class")
    print("Cols = Predicted class\n")

    print("     " + " ".join(f"{i:5d}" for i in range(10)))

    for i, row in enumerate(confusion):
        print(f"{i:3d} | " + " ".join(f"{int(v):5d}" for v in row))
