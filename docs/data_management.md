# data_management/ — Manejo de Datos

Todo lo que toca los datos del TP vive acá.
El resto del proyecto importa de acá y no accede a los CSVs directamente.

## `dataset.py` — Dataset

Dataclass simple con dos campos:
- `X`: matriz de entradas, shape `(n_muestras, n_features)`
- `zeta`: salidas esperadas ζ, shape `(n_muestras,)` o `(n_muestras, n_clases)`

## `loader.py` — Carga de CSV

```python
dataset = load_csv("data/transactions.csv", target_column="isFraud")
```

Lee el CSV con pandas, separa features de target, devuelve un `Dataset`.

## `preprocessing.py` — Preprocesamiento

Tres funciones, cada una recibe `X` y devuelve `X` transformado:

- `normalize(X)` → escala a [0, 1]
- `standardize(X)` → media 0, desviación 1
- `one_hot_encode(zeta, n_classes)` → convierte etiquetas enteras a vectores binarios

**Importante:** el preprocesamiento se fitea solo sobre train y se aplica a val y test.
Nunca fitear sobre val o test (data leakage).

## `splitter.py` — Split de Datos

```python
train, val, test = train_val_test_split(dataset, train=0.7, val=0.15, test=0.15, seed=42)
folds = k_fold_split(dataset, k=5, seed=42)
```
