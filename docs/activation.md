# activation/ — Función de Activación θ(h)

## ABC: `ActivationFunction`

Tres métodos:
- `compute(h: Array) → Array` — aplica θ: calcula `O = θ(h)`
- `derivative(h: Array) → Array` — calcula `θ'(h)`, necesario para δ en backprop
- `is_differentiable() → bool` — retorna `True` por defecto; `StepActivation` retorna `False`

`NeuronLayer.backward` usa `is_differentiable()` para decidir si aplicar la derivada
o usar la regla de Rosenblatt (θ'(h) = 1).

## Implementaciones

| Archivo | Fórmula | Derivada | `is_differentiable()` |
|---|---|---|---|
| `step.py` | θ(h) = 1 si h ≥ 0, sino 0 | N/A | `False` |
| `identity.py` | θ(h) = h | θ'(h) = 1 (ones_like) | `True` |
| `tanh.py` | θ(h) = tanh(β·h) | θ'(h) = β(1 - θ²(h)) | `True` |
| `logistic.py` | θ(h) = 1/(1+e^{-2βh}) | θ'(h) = 2β·θ(h)·(1 - θ(h)) | `True` *(stub)* |
| `relu.py` | θ(h) = max(0, h) | θ'(h) = 1 si h > 0, sino 0 | `True` *(stub)* |

## Parámetro β

`TanhActivation(beta=1.0)` — controla la pendiente de la función.
La derivada de tanh implementada es `β * (1 - tanh²(β·h))`, computada directamente
sobre `h` (no sobre el resultado guardado en el forward).

## Cuándo usar cada una

- **Step**: perceptrón escalón (clasificación binaria, sin backprop real)
- **Identity**: perceptrón lineal (ADALINE, regresión)
- **Tanh**: perceptrón no lineal, capas ocultas del MLP
- **Logistic / ReLU**: pendientes de implementación
