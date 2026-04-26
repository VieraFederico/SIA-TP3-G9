# metric/ — Métricas de Evaluación

## ABC: `Metric`

- `compute(zeta, O)` → float
- `name()` → str (para logging y reportes)

## Implementaciones

| Archivo | Métrica | Cuándo usar |
|---|---|---|
| `accuracy.py` | Accuracy = correctas / total | Clasificación balanceada |
| `precision.py` | TP / (TP + FP) | Cuando los falsos positivos son costosos |
| `recall.py` | TP / (TP + FN) | Cuando los falsos negativos son costosos (ej: fraude) |
| `f1.py` | 2 · (P · R) / (P + R) | Balance precision/recall |
| `mse_metric.py` | (1/N) Σ (ζ - O)² | Regresión |

## Uso

Las métricas se pasan al `Trainer` en el constructor. Al llamar `evaluate()`,
el trainer las calcula todas y devuelve un dict `{nombre: valor}`.

Se configuran desde el JSON o CLI con el campo `metrics`.
