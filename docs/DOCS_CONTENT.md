# Contenido de docs/ — Para Claude Code

Crear cada uno de estos archivos en la carpeta `docs/`.

---

## `docs/network.md`

```markdown
# network/ — La Red Neuronal

Tres clases con relación de composición directa.

## Clases

### `Neuron` (`neuron.py`)
Una sola neurona. Recibe N entradas y devuelve una salida.

**Qué hace:**
1. Calcula la excitación: `h = Σ xᵢwᵢ + w₀`
2. Aplica la función de activación: `O = θ(h)`
3. Guarda `h` para usarlo en backpropagation

**Parámetros del constructor:**
- `n_inputs`: número de entradas
- `activation`: instancia de `ActivationFunction`

### `NeuronLayer` (`neuron_layer.py`)
Una capa de N neuronas, implementada de forma vectorizada con NumPy.
Equivale a una fila de `Neuron`s pero eficiente.

**Qué hace:**
- `forward(x)`: calcula `h = X·W + b`, luego `V = θ(h)`. Guarda `h` y `V`.
- `backward(delta)`: recibe `δ` de la capa siguiente, calcula `delta_w` y `δ` para la capa anterior.

**Por qué guarda h y V:**
Durante el backward necesitamos `θ'(h)` para calcular `δ = δ_siguiente · θ'(h)`.

### `MultilayerPerceptron` (`multilayer_perceptron.py`)
Lista de `NeuronLayer`. Orquesta el forward y backpropagation completos.

**Un perceptrón simple es un caso especial:**
`MultilayerPerceptron` con una sola `NeuronLayer` de una neurona.
La activación (Step, Identity, Tanh...) se inyecta al construirlo.

**Forward pass:** `x → V¹ → V² → ... → O`

**Backpropagation:** propaga `δ` capa por capa de derecha a izquierda
usando la regla de la cadena: `∂E/∂w = δ · V_anterior`

## Relación con la teoría

```
Clase             Símbolo teoría
─────────────────────────────────
Neuron.h          h (excitación)
Neuron.O          O (salida)
NeuronLayer.V     V (salida capa oculta)
NeuronLayer.W     W (pesos de la capa)
delta             δ = (ζ - O)·θ'(h)
```
```

---

## `docs/activation.md`

```markdown
# activation/ — Función de Activación θ(h)

## ABC: `ActivationFunction`

Dos métodos obligatorios:
- `compute(h)` → `O = θ(h)`
- `derivative(h)` → `θ'(h)`, necesario para `δ` en backprop

## Implementaciones

| Archivo | Fórmula | Derivada | Diferenciable |
|---|---|---|---|
| `step.py` | θ(h) = 1 si h ≥ 0, sino 0 | — | No |
| `identity.py` | θ(h) = h | θ'(h) = 1 | Sí |
| `tanh.py` | θ(h) = tanh(β·h) | θ'(h) = β(1 - θ²(h)) | Sí |
| `logistic.py` | θ(h) = 1/(1+e^{-2βh}) | θ'(h) = 2β·θ(h)·(1 - θ(h)) | Sí |
| `relu.py` | θ(h) = max(0, h) | θ'(h) = 1 si h > 0, sino 0 | Sí* |

*ReLU no es diferenciable en h=0, se usa la subderivada.

## Parámetro β

`TanhActivation` y `LogisticActivation` reciben `beta` en el constructor.
Controla la pendiente de la función. Default: `beta=1.0`.

## Cuándo usar cada una

- **Step**: perceptrón escalón (clasificación binaria, no usa backprop)
- **Identity**: perceptrón lineal (ADALINE, regresión)
- **Tanh / Logistic**: perceptrón no lineal, capas ocultas del MLP
- **ReLU**: opcional, capas ocultas de MLPs profundos
```

---

## `docs/cost.md`

```markdown
# cost/ — Función de Costo E(O)

La función de costo mide qué tan lejos está la salida obtenida O de la esperada ζ.
Es lo que el entrenamiento minimiza ajustando los pesos w.

## ABC: `CostFunction`

- `compute(zeta, O)` → `E` (error escalar, un número)
- `gradient(zeta, O)` → `∂E/∂O` (vector, arranca la retropropagación)

## Implementaciones

### `MSE` (`mse.py`)
Error cuadrático medio.
```
E = (1/2) Σ (ζ - O)²
∂E/∂O = -(ζ - O)
```
Usar para: regresión, perceptrón lineal y no lineal del Ej1.

### `BinaryCrossEntropy` (`binary_cross_entropy.py`)
Para clasificación binaria con salida sigmoide.

### `CategoricalCrossEntropy` (`categorical_cross_entropy.py`)
Para clasificación multiclase con salida softmax.
Usar para: MLP de dígitos (Ej2, Ej3).

## Relación con la teoría

`gradient()` devuelve `∂E/∂O`, que es el primer factor de la regla de la cadena
en backpropagation: `∂E/∂w = (∂E/∂O) · (∂O/∂h) · (∂h/∂w)`
```

---

## `docs/optimizer.md`

```markdown
# optimizer/ — Regla de Actualización Δw

El optimizador define cómo actualizar los pesos w dado el gradiente ∂E/∂w.

## ABC: `Optimizer`

- `update(weights, grads)` → lista de pesos actualizados
- `reset()` → limpia el estado interno (velocidades, momentos)

El optimizador mantiene **estado interno entre llamadas** (velocidades en Momentum,
momentos en Adam). Por eso necesita `reset()` entre experimentos.

## Implementaciones

### `GradientDescent` (`gradient_descent.py`)
Descenso por gradiente estándar.
```
Δw = -η · ∂E/∂w
w_nuevo = w + Δw
```
Parámetros: `eta` (η).

### `Momentum` (`momentum.py`)
Acumula una velocidad entre épocas para acelerar la convergencia.
```
v = β·v_anterior + (1-β)·∂E/∂w
w_nuevo = w - η·v
```
Parámetros: `eta` (η), `momentum_beta` (β, default 0.9).

### `Adam` (`adam.py`)
Adapta η por parámetro usando estimaciones de primer y segundo momento.
Parámetros: `eta`, `adam_beta1` (β₁=0.9), `adam_beta2` (β₂=0.999).

## Cuándo usar cada uno

- **GradientDescent**: baseline, simple, para comparar
- **Momentum**: converge más rápido que GD en la mayoría de los casos
- **Adam**: generalmente el mejor para MLP, adapta η automáticamente
```

---

## `docs/metric.md`

```markdown
# metric/ — Métricas de Evaluación

## ABC: `Metric`

- `compute(zeta, O)` → float
- `name()` → str (para logging y reportes)

## Implementaciones

| Archivo | Métrica | Cuándo usar |
|---|---|---|
| `accuracy.py` | Accuracy = correctas / total | Clasificación balanceada |
| `precision.py` | TP / (TP + FP) | Cuando los falsos positivos son costosos |
| `recall.py` | TP / (TP + FN) | Cuando los falsos negativos son costosos (ej: fraude) |
| `f1.py` | 2 · (P · R) / (P + R) | Balance precision/recall |
| `mse_metric.py` | (1/N) Σ (ζ - O)² | Regresión |

## Uso

Las métricas se pasan al `Trainer` en el constructor. Al llamar `evaluate()`,
el trainer las calcula todas y devuelve un dict `{nombre: valor}`.

Se configuran desde el JSON o CLI con el campo `metrics`.
```

---

## `docs/trainer.md`

```markdown
# trainer.py — Loop de Entrenamiento

`Trainer` tiene **una sola responsabilidad**: dado un modelo, datos y config, entrena.
No sabe qué ejercicio es ni de dónde vienen los datos.

## Interfaz

```python
trainer = Trainer(cost_fn, optimizer, metrics, cfg)
history = trainer.fit(model, X_train, zeta_train, X_val, zeta_val)
results = trainer.evaluate(model, X_test, zeta_test)
```

## Lo que hace `fit()`

Por cada época:
1. Recorre los datos según `cfg.training_mode`:
   - **online**: procesa un dato μ a la vez, actualiza w después de cada uno
   - **batch**: procesa todos los datos, actualiza w una vez al final
   - **minibatch**: procesa lotes de `cfg.batch_size`, actualiza w por lote
2. Por cada paso: forward → calcula E → backprop → acumula Δw → `optimizer.update()`
3. Calcula E sobre val para detectar sobreajuste
4. Chequea convergencia: si `E < epsilon` o se llegó a `epochs`, para

## Lo que devuelve `fit()`

```python
{
  "train_error": [0.45, 0.32, 0.21, ...],   # E por época en train
  "val_error":   [0.48, 0.35, 0.24, ...],   # E por época en val
  "epochs": 237                              # épocas hasta converger
}
```

## Relación con `experiments/`

`Trainer` no llama a los ejercicios. Los ejercicios llaman al `Trainer`.
Los flujos de cada ejercicio viven en `src/experiments/`.
```

---

## `docs/experiments.md`

```markdown
# experiments/ — Flujos por Ejercicio

Cada archivo orquesta un ejercicio completo del TP.
Usan `Trainer`, `data_management` y `persistence` como dependencias.

## Interfaz común

Cada archivo expone una función `run(cfg: ExperimentConfig) -> None`.
`main.py` la llama según `--ejercicio`.

## `ejercicio_1.py` — Detección de Fraude

**Dataset:** `transactions.csv`
**Modelo:** perceptrón simple (MLP con 1 neurona)
**Objetivo:** comparar perceptrón lineal vs no lineal, analizar generalización

Flujo:
1. Carga y estandariza `transactions.csv`
2. Split train/val/test
3. Entrena con activación `identity` (lineal)
4. Entrena con activación `tanh` / `logistic` (no lineal)
5. Compara métricas de ambos en test
6. Guarda ambas runs

## `ejercicio_2.py` — Reconocimiento de Dígitos

**Dataset:** `digits.csv`
**Modelo:** MLP configurable
**Objetivo:** explorar efecto de η, arquitectura y optimizador

Flujo:
1. Carga y estandariza `digits.csv`
2. Construye el MLP según `cfg.architecture`
3. Entrena con la config dada (η, optimizer, training_mode)
4. Evalúa accuracy y F1 en test
5. Guarda la run

Para comparar variantes: correr múltiples veces con distintos flags CLI.

## `ejercicio_3.py` — Más Dígitos (objetivo ≥ 98%)

**Dataset:** `more_data_digits.csv` + `digits_test.csv`
**Modelo:** MLP tuneado
**Objetivo:** accuracy ≥ 98% en test

Igual que ej2 pero con más datos de entrenamiento y evaluación en `digits_test.csv`.
```

---

## `docs/data_management.md`

```markdown
# data_management/ — Manejo de Datos

Todo lo que toca los datos del TP vive acá.
El resto del proyecto importa de acá y no accede a los CSVs directamente.

## `dataset.py` — Dataset

Dataclass simple con dos campos:
- `X`: matriz de entradas, shape `(n_muestras, n_features)`
- `zeta`: salidas esperadas ζ, shape `(n_muestras,)` o `(n_muestras, n_clases)`

## `loader.py` — Carga de CSV

```python
dataset = load_csv("data/transactions.csv", target_column="isFraud")
```

Lee el CSV con pandas, separa features de target, devuelve un `Dataset`.

## `preprocessing.py` — Preprocesamiento

Tres funciones, cada una recibe `X` y devuelve `X` transformado:

- `normalize(X)` → escala a [0, 1]
- `standardize(X)` → media 0, desviación 1
- `one_hot_encode(zeta, n_classes)` → convierte etiquetas enteras a vectores binarios

**Importante:** el preprocesamiento se fitea solo sobre train y se aplica a val y test.
Nunca fitear sobre val o test (data leakage).

## `splitter.py` — Split de Datos

```python
train, val, test = train_val_test_split(dataset, train=0.7, val=0.15, test=0.15, seed=42)
folds = k_fold_split(dataset, k=5, seed=42)
```
```

---

## `docs/analysis.md`

```markdown
# analysis/ — Librería de Análisis

**Regla de oro:** `analysis/` solo lee archivos de `results/`.
Nunca importa de `src/trainer.py` ni de `src/network/`.

Esto garantiza que podés analizar runs viejas sin tener el pipeline cargado.

## `data.py` — Leer Resultados

```python
from analysis.data import load_history, load_metrics, compare_runs

history  = load_history("ej2_adam_20260421")   # dict con train_error, val_error, epochs
metrics  = load_metrics("ej2_adam_20260421")   # dict con accuracy, f1, etc.
df       = compare_runs(["run_sgd", "run_momentum", "run_adam"])  # DataFrame pandas
```

## `plots.py` — Graficar

```python
from analysis.plots import plot_error_curve, plot_confusion_matrix, plot_metric_comparison

plot_error_curve(history, save_to="figures/ej2_error.png")
plot_confusion_matrix(zeta, O, save_to="figures/ej2_confusion.png")
plot_metric_comparison(df, metric="accuracy", save_to="figures/ej2_comparison.png")
```

Todas las funciones tienen `save_to` opcional. Si es `None`, muestra el gráfico interactivo.

## Uso típico desde un notebook

```python
import sys
sys.path.append("..")

from analysis.data import compare_runs
from analysis.plots import plot_metric_comparison

# Comparar todos los optimizadores del ej2
df = compare_runs([
    "ej2_sgd_lr001",
    "ej2_momentum_lr001",
    "ej2_adam_lr001",
])
plot_metric_comparison(df, metric="accuracy")
```
```

---

## `docs/utils.md`

```markdown
# utils/ — Utilidades

## `parser.py` — Parser de Argumentos

Define todos los flags CLI del proyecto y resuelve la prioridad:
**CLI sobreescribe JSON, JSON es el default.**

### Uso

```bash
python main.py --ejercicio <1|2|3> --config <path.json> [flags opcionales]
```

### Tabla de flags

| Flag | Tipo | Descripción |
|---|---|---|
| `--ejercicio` | int | **Obligatorio.** 1, 2 o 3. |
| `--config` | str | **Obligatorio.** Path al JSON base. |
| `--name` | str | Nombre de la run. |
| `--seed` | int | Semilla aleatoria. |
| `--data-path` | str | Path al CSV. |
| `--target-column` | str | Columna objetivo del CSV. |
| `--preprocessing` | str | `normalize` / `standardize` / `one_hot` |
| `--split-train` | float | Proporción train. |
| `--split-val` | float | Proporción val. |
| `--split-test` | float | Proporción test. |
| `--activation` | str | `step` / `identity` / `tanh` / `logistic` / `relu` |
| `--beta` | float | β para tanh y logistic. |
| `--architecture` | int+ | Neuronas por capa. Ej: `784 128 10` |
| `--cost-function` | str | `mse` / `binary_cross_entropy` / `categorical_cross_entropy` |
| `--optimizer` | str | `gradient_descent` / `momentum` / `adam` |
| `--eta` | float | Tasa de aprendizaje η. |
| `--momentum-beta` | float | β de Momentum. |
| `--adam-beta1` | float | β₁ de Adam. |
| `--adam-beta2` | float | β₂ de Adam. |
| `--training-mode` | str | `online` / `batch` / `minibatch` |
| `--batch-size` | int | Tamaño del lote (minibatch). |
| `--epochs` | int | Máximo de épocas. |
| `--epsilon` | float | Umbral de convergencia E < ε. |

### Ejemplos

```bash
# Variar optimizador (sin tocar el JSON)
python main.py --ejercicio 2 --config configs/base_ej2.json --optimizer adam --eta 0.001

# Variar arquitectura
python main.py --ejercicio 2 --config configs/base_ej2.json --architecture 784 128 64 10

# Variar learning rate
python main.py --ejercicio 2 --config configs/base_ej2.json --eta 0.1
python main.py --ejercicio 2 --config configs/base_ej2.json --eta 0.01
python main.py --ejercicio 2 --config configs/base_ej2.json --eta 0.001
```

### Cómo funciona internamente

1. Lee `--config` y carga el JSON como `ExperimentConfig`
2. Por cada flag CLI que el usuario pasó **explícitamente**, sobreescribe el campo correspondiente
3. Devuelve `(ejercicio: int, cfg: ExperimentConfig)` listo para usar
```
```
