# TP3 — Perceptrón Simple y Multicapa

Implementation of Simple Perceptron and Multilayer Perceptron (MLP) from scratch using NumPy.

## Setup

```bash
pip install -r requirements.txt
```

## Running experiments

From the **project root**, use `run.py`:

**Short flags** (exercise + case name):

```bash
python run.py -ex 0 -c xor
python run.py -ex 1 -c nonlinear
python run.py -ex 2 -c baseline
python run.py -ex 3 -c tuned
```

Same flags work on the script that holds the training loop:

```bash
python scripts/run_experiment.py -ex 2 -c baseline
```

**Explicit JSON path** (ignores `-ex`/`-c`):

```bash
python run.py --config experiments_configs/ex2_digits/baseline.json
```

See all presets and examples:

```bash
python run.py --help
```

Plots and comparisons are done **outside** this script: import `analysis.loaders` and `analysis.plots` from a notebook or ad-hoc script (see `ARCHITECTURE.md`).

## Running tests

```bash
pytest tests/
```

## Project structure

See `ARCHITECTURE.md` for module layout, `-ex`/`-c` mapping (`utils/experiment_cli.py`), and how JSON relates to `src/config.py`.
