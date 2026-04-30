# experiments/ — Flujos por Ejercicio y Tutoriales

Todos los experimentos viven en `src/experiments/`. `main.py` los enruta según el flag CLI.

## Tutoriales (acceso: `--tutorial {step,linear,nonlinear,mlp}`)

Los tutoriales no necesitan `--config` ni datos externos. Sirven para verificar que
forward, backward y el optimizer funcionan correctamente.

### `tutorial_step.py` — Compuerta AND ✓
Arquitectura: `MultilayerPerceptron([NeuronLayer(2, 1, StepActivation())])`
- Entrena con `eta=0.1`, `epochs=50`, `training_mode="online"`
- Resultado esperado: predicciones `[0, 0, 0, 1]`
- Salida: `output/tutorial/step/` — frontera de decisión + curva de error

### `tutorial_linear.py` — ADALINE ✓
Arquitectura: `MultilayerPerceptron([NeuronLayer(1, 1, IdentityActivation())])`
- Ajusta `y = ax + b` con MSE
- Salida: `output/tutorial/linear/` — regresión + curva de error

### `tutorial_non_linear.py` — Tanh simple ✓
Arquitectura: `MultilayerPerceptron([NeuronLayer(1, 1, TanhActivation(β))])`
- Aproxima `tanh(x)` con un perceptrón de 1 neurona
- Salida: `output/tutorial/nonlinear/`

### `tutorial_mlp_tanh.py` — MLP [1→5→1] ✓
Arquitectura: `MultilayerPerceptron([NeuronLayer(1, 5, TanhActivation()), NeuronLayer(5, 1, TanhActivation())])`
- Aproxima `tanh(x)` en [-5, 5] con `eta=0.05`, `epochs=500`
- Verifica que forward y backward a través de múltiples capas funcionen
- Salida: `output/tutorial/mlp_tanh/` — regresión + curva de error

## Ejercicios (acceso: `--ejercicio {1,2,3} --config <path.json>`)

Cada archivo expone `run(cfg: ExperimentConfig) -> None`. Todos están pendientes.

### `ejercicio_1.py` — Detección de Fraude *(stub)*
**Config base:** `configs/base_ej1.json`
- Dataset: `data/transactions.csv`, target `isFraud`
- Arquitectura: `[10, 1]`, activación tanh
- Preprocesamiento: standardize, split 70/15/15
- Objetivo: comparar perceptrón lineal (identity) vs no lineal (tanh/logistic)

### `ejercicio_2.py` — Reconocimiento de Dígitos *(stub)*
**Config base:** `configs/base_ej2.json`
- Dataset: `data/digits.csv`, target `label`
- Arquitectura: `[784, 64, 10]`, costo `categorical_cross_entropy`
- Preprocesamiento: standardize, split 80/10/10, minibatch 32
- Objetivo: explorar efecto de η, arquitectura y optimizador

### `ejercicio_3.py` — Dígitos con objetivo ≥ 98% *(stub)*
**Config base:** `configs/base_ej3.json`
- Dataset: `data/more_data_digits.csv`
- Arquitectura: `[784, 128, 64, 10]`, optimizer Adam, `eta=0.001`
- Igual que ej2 pero con más datos y arquitectura más profunda
