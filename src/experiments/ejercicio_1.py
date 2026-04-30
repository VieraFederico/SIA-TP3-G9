from cost.mse import MSECost
from network.multilayer_perceptron import MultilayerPerceptron
from optimizer.gradient_descent import GradientDescent
from src.config import ExperimentConfig
from src.data_management.loader import load_csv
from network.neuron_layer import NeuronLayer
from activation.identity import IdentityActivation
from trainer import Trainer


def run(cfg: ExperimentConfig) -> None:
    """Ejercicio 1 — Detección de fraude.

    Entrena perceptrón simple lineal vs no lineal. Compara generalización.
    """
    #TODO: LEER ESTO. para el ejercicio 1 viendo el dataset, uno puede observar un par de patrones:
    # 1. Las compras que suelen ser de mayor valor, suelen ser fraudulentas.
    # 2. Las compras que se realizan en poco tiempo de sesion, suelen ser fraudulentas.
    # 3. Las compras con cuentas de pocos dias de uso
    # 4. La diferencia de dias entre las compras suelen ser menor
    # Aca uno se pregunta, si necesita usar las columnas MAS relevantes, o directamente usar todas.
    # Probemos con usar todas a ver como sale *excepto las ultimas, que es lo que quiero estimar*.

    target_columns = "big_model_fraud_probability"

    dataset = load_csv(cfg.data_path, target_columns=target_columns)
    train_dataset,val_dataset, test_dataset  = dataset.split(dataset)

    model = MultilayerPerceptron([
        NeuronLayer(n_inputs=1, n_neurons=1, activation=IdentityActivation()),
    ])


    trainer = Trainer(
        cost_fn=MSECost(),
        optimizer=GradientDescent(learning_rate=cfg.eta),
        metrics=[],
        cfg=cfg,
    )
