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
