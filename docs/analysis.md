# analysis/ — Librería de Análisis

**Regla de oro:** `analysis/` solo lee archivos de `results/` y no importa de `src/`.
Esto garantiza que podés analizar runs viejas sin tener el pipeline cargado.

## `plots.py` — Graficar

### `plot_error_curve(history, output_path)` ✓
Grafica E vs épocas (train y val). Guarda siempre en disco (no tiene modo interactivo).

```python
from analysis.plots import plot_error_curve
plot_error_curve(history, output_path="output/tutorial/step/error.png")
```

`history` es el dict que devuelve `trainer.fit()` — necesita las claves
`"train_error"` y `"val_error"`.

Si `val_error` sube mientras `train_error` baja → overfitting.

### `plot_decision_boundary(X, zeta, model, title, output_path)` ✓
Grafica la frontera de decisión de un clasificador binario.

```python
from analysis.plots import plot_decision_boundary
plot_decision_boundary(X, zeta, modelo, title="AND", output_path="output/boundary.png")
```

- `X`: matriz `(n, 2)` — solo funciona con 2 features
- Crea una grilla 300×300, predice la clase de cada punto y colorea el fondo
- La frontera visible es el hiperplano `w₁x₁ + w₂x₂ + w₀ = 0`

### `plot_regression(X, zeta, model, title, output_path, xlim=(-6,6), ylim=(-1.5,1.5))` ✓
Grafica la curva predicha vs los datos reales. Usado para ADALINE y tanh.

```python
from analysis.plots import plot_regression
plot_regression(X, zeta, modelo, title="ADALINE", output_path="output/regression.png")
```

### `plot_confusion_matrix(zeta, O, save_to=None)` *(stub)*
### `plot_metric_comparison(df, metric, save_to=None)` *(stub)*

## `data.py` — Leer Resultados *(todo stub)*

```python
from analysis.data import load_history, load_metrics, compare_runs
```

Todas las funciones están pendientes de implementación. Están pensadas para leer
archivos guardados por `src/persistence.py` en `results/`.
