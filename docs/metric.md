# metric/ — Métricas de Evaluación

Todas las métricas están pendientes de implementación (stubs con `NotImplementedError`).

## ABC: `Metric`

- `compute(zeta, O) → float`
- `name() → str` — para logging y reportes

## Implementaciones *(todas stub)*

| Archivo | Métrica | Cuándo usar |
|---|---|---|
| `accuracy.py` | correctas / total | Clasificación balanceada |
| `precision.py` | TP / (TP + FP) | Cuando los falsos positivos son costosos |
| `recall.py` | TP / (TP + FN) | Cuando los falsos negativos son costosos (ej: fraude) |
| `f1.py` | 2 · P · R / (P + R) | Balance precision/recall |
| `mse_metric.py` | (1/N) Σ (ζ - O)² | Regresión |

## Uso

Las métricas se pasan al `Trainer` en el constructor.
`trainer.evaluate(model, X_test, zeta_test)` las calcula todas y devuelve
un dict `{metric.name(): valor}`.

Se configuran en el JSON con el campo `"metrics"`:
```json
"metrics": ["accuracy", "f1"]
```

Las métricas disponibles en `base_ej1.json`: `["mse", "accuracy"]`
Las métricas disponibles en `base_ej2.json` y `base_ej3.json`: `["accuracy", "f1"]`
