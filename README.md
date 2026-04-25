# TP3 — Redes Neuronales: Perceptrón Simple y Multicapa

Implementación from scratch de perceptrón simple y MLP para los ejercicios del TP3.
Todo el código usa NumPy vectorizado. Sin frameworks de ML.

## Estructura

```
src/            Código principal (red, entrenamiento, datos)
analysis/       Librería de análisis y gráficos (solo lee results/)
utils/          Parser de argumentos CLI
configs/        JSONs base por ejercicio
results/        Runs guardadas (generado al correr)
docs/           Documentación de cada módulo
```

## Cómo correr

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejercicio 1 — Fraude (perceptrón simple)
python main.py --ejercicio 1 --config configs/base_ej1.json

# Ejercicio 2 — Dígitos (MLP)
python main.py --ejercicio 2 --config configs/base_ej2.json

# Ejercicio 3 — Más dígitos
python main.py --ejercicio 3 --config configs/base_ej3.json
```

## Sobreescribir hiperparámetros desde CLI

Cualquier parámetro del JSON puede sobreescribirse sin editar el archivo:

```bash
# Cambiar optimizer y learning rate
python main.py --ejercicio 2 --config configs/base_ej2.json --optimizer adam --eta 0.001

# Cambiar arquitectura
python main.py --ejercicio 2 --config configs/base_ej2.json --architecture 784 128 64 10

# Cambiar modo de entrenamiento
python main.py --ejercicio 2 --config configs/base_ej2.json --training-mode batch
```

Ver la tabla completa de flags en `docs/utils.md`.

## Resultados

Cada run genera un directorio en `results/<run_id>/`:

```
results/ej2_base_20260421_143022/
├── config.json     # Config exacta que se usó
├── weights.npz     # Pesos entrenados
├── history.json    # Error E por época
└── metrics.json    # Métricas finales
```

## Análisis

```python
from analysis.data import load_history, compare_runs
from analysis.plots import plot_error_curve, plot_metric_comparison

# Graficar curva de error de una run
history = load_history("ej2_base_20260421_143022")
plot_error_curve(history, save_to="figures/ej2_error.png")

# Comparar múltiples runs
df = compare_runs(["run_sgd", "run_momentum", "run_adam"])
plot_metric_comparison(df, metric="accuracy")
```

## Correspondencia teoría → código

| Teoría | Símbolo | Código |
|---|---|---|
| Excitación | h = Σ xᵢwᵢ | `h` |
| Función de activación | θ(h) | `ActivationFunction` |
| Salida obtenida | O = θ(h) | `O` |
| Salida esperada | ζ | `zeta` |
| Función de costo | E(O) | `CostFunction` |
| Delta backprop | δ | `delta` |
| Tasa de aprendizaje | η | `eta` |
| Actualización de pesos | Δw | `delta_w` |
| Convergencia | E < ε | `epsilon` |

## Documentación

Ver `docs/` para la descripción detallada de cada módulo.
