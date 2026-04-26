# experiments/ — Flujos por Ejercicio

Cada archivo orquesta un ejercicio completo del TP.
Usan `Trainer`, `data_management` y `persistence` como dependencias.

## Interfaz común

Cada archivo expone una función `run(cfg: ExperimentConfig) -> None`.
`main.py` la llama según `--ejercicio`.

## `ejercicio_1.py` — Detección de Fraude

**Dataset:** `transactions.csv`
**Modelo:** perceptrón simple (MLP con 1 neurona)
**Objetivo:** comparar perceptrón lineal vs no lineal, analizar generalización

Flujo:
1. Carga y estandariza `transactions.csv`
2. Split train/val/test
3. Entrena con activación `identity` (lineal)
4. Entrena con activación `tanh` / `logistic` (no lineal)
5. Compara métricas de ambos en test
6. Guarda ambas runs

## `ejercicio_2.py` — Reconocimiento de Dígitos

**Dataset:** `digits.csv`
**Modelo:** MLP configurable
**Objetivo:** explorar efecto de η, arquitectura y optimizador

Flujo:
1. Carga y estandariza `digits.csv`
2. Construye el MLP según `cfg.architecture`
3. Entrena con la config dada (η, optimizer, training_mode)
4. Evalúa accuracy y F1 en test
5. Guarda la run

Para comparar variantes: correr múltiples veces con distintos flags CLI.

## `ejercicio_3.py` — Más Dígitos (objetivo ≥ 98%)

**Dataset:** `more_data_digits.csv` + `digits_test.csv`
**Modelo:** MLP tuneado
**Objetivo:** accuracy ≥ 98% en test

Igual que ej2 pero con más datos de entrenamiento y evaluación en `digits_test.csv`.
