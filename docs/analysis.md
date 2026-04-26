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
