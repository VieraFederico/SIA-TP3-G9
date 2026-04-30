import numpy as np
from src.network.multilayer_perceptron import MultilayerPerceptron
from src.network.neuron_layer import NeuronLayer
from src.activation.tanh import TanhActivation
from src.cost.mse import MSECost
from src.optimizer.gradient_descent import GradientDescent
from src.trainer import Trainer
from src.config import ExperimentConfig
from analysis.plots import plot_error_curve


def run() -> None:
    """Aproximación de tanh(x) con un MLP real de dos capas [1 → 5 → 1].

    tutorial_non_linear usaba un único Neuron — aquí usamos MultilayerPerceptron
    con una capa oculta de 5 neuronas tanh seguida de una capa de salida de 1 neurona tanh.
    Sirve para verificar que forward y backpropagation a través de múltiples capas funcionen.

    Resultado esperado:
        La curva predicha se superpone con tanh(x) en [-5, 5], convergiendo
        más rápido que el perceptrón simple gracias a la capacidad extra.
    """

    # ── DATOS ────────────────────────────────────────────────────────────
    X_flat = np.linspace(-5, 5, 50)
    zeta   = np.tanh(X_flat)
    X      = X_flat.reshape(-1, 1)

    # ── MODELO ───────────────────────────────────────────────────────────
    # Arquitectura [1, 5, 1]: capa oculta de 5 neuronas tanh, salida de 1 neurona tanh
    modelo = MultilayerPerceptron([
        NeuronLayer(n_inputs=1, n_neurons=5, activation=TanhActivation(beta=1.0)),
        NeuronLayer(n_inputs=5, n_neurons=1, activation=TanhActivation(beta=1.0)),
    ])

    # ── CONFIG ───────────────────────────────────────────────────────────
    cfg = ExperimentConfig(
        name="tutorial_mlp_tanh",
        seed=42,
        data_path="",
        target_column="",
        preprocessing="normalize",
        split_train=1.0,
        split_val=0.0,
        split_test=0.0,
        activation="tanh",
        beta=1.0,
        architecture=[1, 5, 1],
        cost_function="mse",
        optimizer="gradient_descent",
        eta=0.05,
        training_mode="online",
        epochs=500,
        epsilon=1e-4,
    )

    # ── TRAINER ──────────────────────────────────────────────────────────
    trainer = Trainer(
        cost_fn=MSECost(),
        optimizer=GradientDescent(learning_rate=cfg.eta),
        metrics=[],
        cfg=cfg,
    )

    history = trainer.fit(
        modelo,
        X_train=X, zeta_train=zeta,
        X_val=X,   zeta_val=zeta,
    )

    # ── RESULTADOS ───────────────────────────────────────────────────────
    predictions = np.array([modelo.forward(xi).item() for xi in X])
    final_error = history["train_error"][-1]

    print("=== MLP Tanh [1→5→1] — verificación de arquitectura ===")
    print(f"Épocas:       {history['epochs']}")
    print(f"Error final:  {final_error:.6f}")
    print(f"Error máx |predicción - tanh(x)|: {np.max(np.abs(predictions - zeta)):.4f}")

    # ── GRÁFICO — predicción vs tanh(x) ──────────────────────────────────
    import matplotlib.pyplot as plt
    from pathlib import Path

    out_dir = Path("output/tutorial/mlp_tanh")
    out_dir.mkdir(parents=True, exist_ok=True)

    orden    = np.argsort(X_flat)
    x_sorted = X_flat[orden]

    plt.figure(figsize=(7, 5))
    plt.scatter(X_flat, zeta, color="royalblue", edgecolor="black", s=45, label="tanh(x) real")
    plt.plot(x_sorted, predictions[orden], color="crimson", linewidth=2, label="MLP predicción")
    plt.xlim(-6, 6)
    plt.ylim(-1.5, 1.5)
    plt.axhline(0, color="gray", linewidth=1, alpha=0.6)
    plt.axvline(0, color="gray", linewidth=1, alpha=0.6)
    plt.title("MLP [1→5→1] — aproximación de tanh(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(out_dir / "regression.png", dpi=150, bbox_inches="tight")
    plt.close()

    plot_error_curve(history, output_path=str(out_dir / "error_curve.png"))
    print(f"Gráficos guardados en {out_dir}/")
