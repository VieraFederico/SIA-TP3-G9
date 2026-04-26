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
