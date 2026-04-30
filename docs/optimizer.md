# optimizer/ — Regla de Actualización Δw

El optimizador define cómo actualizar los pesos w dado el gradiente ∂E/∂w.

## ABC: `Optimizer`

- `update(params, grads) → list[tuple[Array, Array]]`
  - Recibe y devuelve listas de tuplas `(weights, bias)`, una por capa
  - El Trainer no necesita aplanar ni reconstruir la estructura del modelo
- `reset()` — limpia el estado interno (velocidades, momentos)

El optimizador puede mantener **estado interno entre llamadas** (velocidades en Momentum,
momentos en Adam). Por eso necesita `reset()` entre experimentos.

## Implementaciones

### `GradientDescent` (`gradient_descent.py`) ✓ Implementado
Descenso por gradiente estándar.
```
w_nuevo = w - learning_rate · ∂E/∂w
b_nuevo = b - learning_rate · ∂E/∂b
```
**Constructor:** `GradientDescent(learning_rate: float)`
El parámetro se llama `learning_rate` (no `eta`). En los tutoriales se instancia como:
```python
GradientDescent(learning_rate=cfg.eta)
```

### `Momentum` (`momentum.py`) *(stub)*
Acumula velocidad entre pasos para acelerar la convergencia.
```
v = β·v_anterior + (1-β)·∂E/∂w
w_nuevo = w - η·v
```
Parámetros: `eta` (η), `momentum_beta` (β, default 0.9 en `ExperimentConfig`).

### `Adam` (`adam.py`) *(stub)*
Adapta η por parámetro usando estimaciones de primer y segundo momento.
Parámetros: `eta`, `adam_beta1` (β₁=0.9), `adam_beta2` (β₂=0.999).

## Cuándo usar cada uno

- **GradientDescent**: baseline, simple, para comparar
- **Momentum**: converge más rápido que GD en la mayoría de los casos
- **Adam**: generalmente el mejor para MLP, adapta η automáticamente (configurado en `base_ej3.json`)
