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
1. Recorre los datos según `cfg.training_mode`:
   - **online**: procesa un dato μ a la vez, actualiza w después de cada uno
   - **batch**: procesa todos los datos, actualiza w una vez al final
   - **minibatch**: procesa lotes de `cfg.batch_size`, actualiza w por lote
2. Por cada paso: forward → calcula E → backprop → acumula Δw → `optimizer.update()`
3. Calcula E sobre val para detectar sobreajuste
4. Chequea convergencia: si `E < epsilon` o se llegó a `epochs`, para

## Lo que devuelve `fit()`

```python
{
  "train_error": [0.45, 0.32, 0.21, ...],   # E por época en train
  "val_error":   [0.48, 0.35, 0.24, ...],   # E por época en val
  "epochs": 237                              # épocas hasta converger
}
```

## Relación con `experiments/`

`Trainer` no llama a los ejercicios. Los ejercicios llaman al `Trainer`.
Los flujos de cada ejercicio viven en `src/experiments/`.
