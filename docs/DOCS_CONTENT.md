# Índice de documentación — SIA TP3 G9

| Archivo | Qué cubre |
|---|---|
| [network.md](network.md) | `Neuron`, `NeuronLayer`, `MultilayerPerceptron` |
| [activation.md](activation.md) | Funciones de activación θ(h) |
| [cost.md](cost.md) | Funciones de costo E(O) |
| [optimizer.md](optimizer.md) | Reglas de actualización Δw |
| [trainer.md](trainer.md) | Loop de entrenamiento genérico |
| [experiments.md](experiments.md) | Ejercicios y tutoriales |
| [data_management.md](data_management.md) | Carga, preprocesamiento y split de datos |
| [metric.md](metric.md) | Métricas de evaluación |
| [analysis.md](analysis.md) | Plots y lectura de resultados |
| [utils.md](utils.md) | CLI — flags y ejemplos |
| [mlp_flow.md](mlp_flow.md) | Walkthrough completo forward/backward |

## Estado de implementación

**Implementado:**
- Activaciones: Step, Identity, Tanh
- Costo: MSE (compute + gradient)
- Red: NeuronLayer, MultilayerPerceptron
- Optimizer: GradientDescent
- Trainer: modo online
- Tutoriales: step (AND), linear (ADALINE), nonlinear (tanh), mlp ([1→5→1])
- Plots: `plot_error_curve`, `plot_decision_boundary`, `plot_regression`

**Pendiente (NotImplementedError):**
- Activaciones: Logistic, ReLU
- Costo: BinaryCrossEntropy, CategoricalCrossEntropy
- Data: load_csv, preprocessing, splitting
- Métricas: todas (accuracy, precision, recall, F1, mse_metric)
- Optimizer: Momentum, Adam
- Trainer: modos batch y minibatch
- Experimentos: ejercicio_1, ejercicio_2, ejercicio_3
- Analysis: load_history, load_metrics, compare_runs, plot_confusion_matrix, plot_metric_comparison
