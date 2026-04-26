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
