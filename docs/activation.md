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
