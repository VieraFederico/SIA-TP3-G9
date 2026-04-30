# trainer.py — Loop de Entrenamiento

`Trainer` tiene **una sola responsabilidad**: dado un modelo, datos y config, entrena.
No sabe qué ejercicio es ni de dónde vienen los datos.

## Interfaz

```python
trainer = Trainer(cost_fn, optimizer, metrics, cfg)
history = trainer.fit(model, X_train, zeta_train, X_val, zeta_val)
results = trainer.evaluate(model, X_test, zeta_test)
```

## Lo que hace `fit()`

Por cada época:
1. Llama al método de entrenamiento según `cfg.training_mode`
2. Registra el error de train (suma de pérdidas por muestra)
3. Registra el error de val con `_evaluate_loss` (sin tocar pesos)
4. Chequea convergencia: si `train_error[-1] < cfg.epsilon` o se llegó a `cfg.epochs`, para

## Modos de entrenamiento

### `online` ✓ Implementado
Actualiza los pesos después de **cada muestra individual**.
```
for cada (xi, zi):
    O    = model.forward(xi)
    grad = cost.gradient(zi, O)
    model.backward(grad)
    model.set_weights(optimizer.update(pesos, gradientes))
    total_loss += cost.compute(zi, O)
```

### `minibatch` *(NotImplementedError)*
Requiere que `NeuronLayer` soporte acumulación de gradientes (`grad_W +=` en lugar de `=`).
Pendiente de implementación.

### `batch` *(NotImplementedError)*
Igual que minibatch pero sobre todos los datos. También pendiente.

## Lo que devuelve `fit()`

```python
{
  "train_error": [0.45, 0.32, 0.21, ...],   # suma de E por muestra, por época
  "val_error":   [0.48, 0.35, 0.24, ...],   # ídem sobre val
  "epochs": 237                              # épocas hasta converger o agotar el límite
}
```

La unidad de `train_error` y `val_error` es la **suma** de `cost.compute(zi, O)` sobre
todas las muestras de la época — no un promedio.

## `evaluate()`

```python
def evaluate(self, model, X, zeta) -> dict[str, float]:
    predictions = np.array([model.forward(xi) for xi in X])
    return {metric.name(): metric.compute(zeta, predictions) for metric in self.metrics}
```

Itera todas las muestras, recolecta predicciones y calcula cada métrica configurada.

## Relación con `experiments/`

`Trainer` no llama a los ejercicios. Los ejercicios llaman al `Trainer`.
Los flujos de cada ejercicio viven en `src/experiments/`.
