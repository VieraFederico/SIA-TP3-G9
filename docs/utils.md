# utils/ — Utilidades

## `parser.py` — Parser de Argumentos

Define todos los flags CLI y resuelve la prioridad:
**CLI sobreescribe JSON, JSON es el default.**

### Uso

```bash
# Correr un ejercicio (requiere --config)
python main.py --ejercicio <1|2|3> --config <path.json> [overrides...]

# Correr un tutorial (no requiere --config)
python main.py --tutorial <step|linear|nonlinear|mlp>
```

`--ejercicio`/`-e` y `--tutorial`/`-t` son **mutuamente excluyentes** y uno es obligatorio.

### Tabla de flags

| Flag | Alias | Tipo | Descripción |
|---|---|---|---|
| `--ejercicio` | `-e` | int `{1,2,3}` | Ejercicio a correr. Requiere `--config`. |
| `--tutorial` | `-t` | str `{step,linear,nonlinear,mlp}` | Tutorial de sanidad. Sin `--config`. |
| `--config` | `-c` | Path | JSON de configuración (obligatorio con `--ejercicio`). |
| `--eta` | | float | Tasa de aprendizaje η. |
| `--epochs` | | int | Máximo de épocas. |
| `--epsilon` | | float | Umbral de convergencia E < ε. |
| `--activation` | | str | `step` / `identity` / `tanh` / `logistic` / `relu` |
| `--beta` | | float | β para tanh/logistic. |
| `--optimizer` | | str | `gradient_descent` / `momentum` / `adam` |
| `--training-mode` | | str | `online` / `batch` / `minibatch` |
| `--batch-size` | | int | Tamaño del lote (minibatch). |
| `--seed` | | int | Semilla aleatoria. |
| `--momentum-beta` | | float | β de Momentum. |
| `--adam-beta1` | | float | β₁ de Adam. |
| `--adam-beta2` | | float | β₂ de Adam. |

**Nota:** los flags de datos (`--data-path`, `--preprocessing`, `--split-train`, etc.)
y de arquitectura (`--architecture`, `--cost-function`) se configuran solo desde el JSON.
No tienen override CLI.

### Ejemplos

```bash
# Tutorial de compuerta AND
python main.py --tutorial step

# Tutorial MLP multicapa
python main.py --tutorial mlp

# Ejercicio 1 con config base
python main.py --ejercicio 1 --config configs/base_ej1.json

# Ejercicio 2 variando optimizer (sobreescribe el JSON)
python main.py --ejercicio 2 --config configs/base_ej2.json --optimizer adam --eta 0.001

# Ejercicio 2 variando learning rate
python main.py --ejercicio 2 --config configs/base_ej2.json --eta 0.1
python main.py --ejercicio 2 --config configs/base_ej2.json --eta 0.01
python main.py --ejercicio 2 --config configs/base_ej2.json --eta 0.001
```

### Cómo funciona internamente

1. Parsea `sys.argv`
2. Si es tutorial: devuelve `("tutorial", nombre, None)` directamente
3. Si es ejercicio: lee `--config` y construye `ExperimentConfig` desde el JSON
4. Por cada flag CLI que se haya pasado explícitamente, sobreescribe el campo del config
5. Devuelve `("ejercicio", número: int, cfg: ExperimentConfig)`

`parse_args()` devuelve `(mode: str, key: int | str, cfg: ExperimentConfig | None)`.
