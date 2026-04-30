# cost/ — Función de Costo E(O)

La función de costo mide qué tan lejos está la salida obtenida O de la esperada ζ.
Es lo que el entrenamiento minimiza ajustando los pesos w.

## ABC: `CostFunction`

- `compute(zeta, O) → float` — error escalar E
- `gradient(zeta, O) → Array` — `∂E/∂O`, arranca la retropropagación

## Implementaciones

### `MSECost` (`mse.py`) ✓ Implementado
Error cuadrático medio, normalizado por el número de salidas N.
```
E(ζ, O)  = (1 / 2N) · Σ (ζᵢ - Oᵢ)²
∂E/∂O    = -(ζ - O) / N
```

Donde N es el tamaño del vector de salida (`len(zeta)` si es array, 1 si es escalar).
Para un perceptrón con 1 salida, N=1 y se reduce a `(1/2)(ζ-O)²`.

Usar para: regresión, perceptrón lineal y no lineal (Ej1), tutoriales.

### `BinaryCrossEntropy` (`binary_cross_entropy.py`) *(stub)*
Para clasificación binaria con salida sigmoide/logistic.

### `CategoricalCrossEntropy` (`categorical_cross_entropy.py`) *(stub)*
Para clasificación multiclase con salida softmax.
Usar para: MLP de dígitos (Ej2, Ej3) — configurado en `base_ej2.json` y `base_ej3.json`.

## Relación con la teoría

`gradient()` devuelve `∂E/∂O`, primer factor de la regla de la cadena:
```
∂E/∂w = (∂E/∂O) · (∂O/∂h) · (∂h/∂w)
           ↑           ↑          ↑
       cost.gradient  θ'(h)    _x (entrada)
```

El `Trainer` pasa este gradiente a `model.backward()` para arrancar la retropropagación.
