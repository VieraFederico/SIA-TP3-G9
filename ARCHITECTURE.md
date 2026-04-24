# TP3 — Perceptrón Simple y Multicapa — Architecture Reference

Developer-oriented description of module responsibilities, dependency rules, config flow, and the `analysis/` boundary. Read this before adding features.

---

## 1. Design principles

| Principle | How it shows up |
|---|---|
| **SRP** | The network does not know about CSVs. The CSV loader does not know about metrics. The trainer does not render plots. |
| **OCP** | New activations, optimizers, losses, metrics, and models are added by subclassing an ABC and registering the class in `src/factory.py`. |
| **DIP** | Trainers depend on `Model`, `LossFunction`, `Optimizer`, `Metric` (ABCs), not concrete MLP / MSE / Adam. |
| **LSP** | Any `ActivationFunction` implementation is interchangeable in the perceptron without breaking the training loop. |
| **Experiment vs analysis** | Training code is isolated from plotting and reporting; analysis only reads files under `experiments/` (recommended by the assignment brief). |
| **KISS** | Plots are plain functions in `analysis/plots.py`, not class hierarchies. |
| **Vectorised math** | NumPy matrix operations everywhere; no Python loops over samples in hot paths. |
| **Reproducibility** | Every run is defined by a JSON under `experiments_configs/`; a snapshot is saved next to outputs. Randomness uses `np.random.Generator` from `src/utils/random_state.py`. |

---

## 2. Config code vs config data

Two different things are both called “config”:

| | What | Where | Who edits |
|---|---|---|---|
| **`src/config.py`** | Code: the `ExperimentConfig` dataclass (the mould). | In `src/`. | When you add a new field to the schema. |
| **`experiments_configs/**/*.json`** | Data: concrete experiments (instances of the mould). | Project root, grouped by exercise. | Whenever you try a new LR, architecture, etc. |

`ExperimentConfig` is the class; each JSON is one instance.

---

## 3. Directory layout

```
<project_root>/
├── README.md
├── requirements.txt
├── run.py                          # Thin entry: parse via utils → run_experiment in scripts/
├── utils/                          # Project-level CLI (not src.utils)
│   ├── experiment_cli.py          # -ex / -c → experiments_configs/... path
│   └── run_parser.py             # argparse + resolve_config_from_args / parse_run_config
├── scripts/
│   └── run_experiment.py         # Main training loop entry (run_experiment)
├── experiments_configs/          # Experiment JSONs (data)
│   ├── validation/
│   ├── ex1_fraud/
│   ├── ex2_digits/
│   └── ex3_more_digits/
├── data/                         # CSV datasets (not always tracked)
├── experiments/                 # Run output (git-ignored)
│   └── <run_id>/
│       ├── config.json
│       ├── model.npz
│       ├── history.json
│       └── metrics.json
├── src/
│   ├── core/                    # ABCs: Model, Layer, ActivationFunction, LossFunction, Optimizer; types
│   ├── activations/, losses/, optimizers/, layers/, models/
│   ├── training/                # Trainer ABC, online/batch/minibatch, stopping, TrainingHistory
│   ├── data/
│   ├── metrics/
│   ├── persistence/             # model_io, ExperimentWriter / ExperimentReader
│   ├── config.py                # ExperimentConfig + load_config / save_config
│   ├── factory.py               # Registries + build_experiment
│   └── utils/
├── analysis/                    # Offline helpers only (see §6)
│   ├── loaders.py
│   └── plots.py
└── tests/
```

---

## 4. Entry points (`run.py` + `scripts/run_experiment.py`)

- **`run.py`** (repo root): inserts the project on `sys.path`, calls `utils.run_parser.parse_run_config`, then `scripts.run_experiment.run_experiment(config_path, run_id)`.
- **`scripts/run_experiment.py`**: owns **`run_experiment`** — the main training/save loop once implemented. Can also be run directly:  
  `python scripts/run_experiment.py -ex 2 -c baseline` (same flags as `run.py`; `prog` in help shows `scripts/run_experiment.py`).

CLI details (argparse, validation):

- **`utils/run_parser.py`**: `build_parser`, `resolve_config_from_args`, `parse_run_config`.
- **`utils/experiment_cli.py`**: preset table and `resolve_config_path`, `describe_cli_help`.

Examples:

```bash
python run.py -ex 2 -c baseline
python run.py --config experiments_configs/ex2_digits/baseline.json
python scripts/run_experiment.py -ex 0 -c xor
```

- **`-ex` / `--exercise`**: `0` = validation, `1` = fraud, `2` = digits, `3` = more digits.
- **`-c` / `--case`**: Short preset — see `python run.py --help`.
- **`--config`**: Explicit JSON path (ignores `-ex`/`-c`).

After training, plots use `analysis/` from a notebook (see §6).

---

## 5. Core abstractions (contracts)

### `ActivationFunction` (`src/core/activation.py`)

 Stateless. `forward(h)`, `derivative(h)` on pre-activation `h = W·x + b`. Optional `is_differentiable()` (e.g. step returns `False`).

### `LossFunction` (`src/core/loss.py`)

`compute(y_true, y_pred) -> float`, `gradient(y_true, y_pred) -> Array` (∂L/∂ŷ).

### `Optimizer` (`src/core/optimizer.py`)

Stateful. `update(params, grads)`, `reset()`.

### `Layer` / `Model` (`src/core/layer.py`, `src/core/model.py`)

Standard forward/backward and flat parameter lists for the optimizer.

### `Metric` (`src/metrics/metric.py`)

`name()`, `compute(y_true, y_pred) -> float`.

### `Trainer` (`src/training/trainer.py`)

Abstract `train_epoch(X, y)`; `BaseTrainer` implements `fit` / `evaluate`. Constructed with model, loss, optimizer, stopping, metrics, logger.

### `StoppingCriterion` (`src/training/stopping.py`)

`should_stop(history: TrainingHistory) -> bool`. Implementations include `MaxEpochs`, `LossThreshold`, `EarlyStopping`, `CompositeStopping`.

---

## 6. The `analysis/` package

**Rules:** analysis only reads artefacts on disk under `experiments/`. It **must not** import `src.training` or `src.models`.

- **`analysis/loaders.py`**: `load_run`, `load_history`, `load_metrics`, `compare_runs`, plus `RunHistory` (JSON-compatible snapshot so analysis does not depend on `TrainingHistory` from training).
- **`analysis/plots.py`**: `plot_loss_curve`, `plot_confusion_matrix`, `plot_metric_comparison`. `plot_decision_boundary` is intentionally not implemented here (would require a live `Model`).

Typical notebook usage:

```python
from analysis.loaders import load_history, compare_runs
from analysis.plots import plot_loss_curve, plot_metric_comparison

history = load_history("run_2026-04-21_xor_mlp")
plot_loss_curve(history, save_to="figures/xor_loss.png")

df = compare_runs(["run_a", "run_b", "run_c"])
plot_metric_comparison(df, metric="accuracy", save_to="figures/cmp.png")
```

---

## 7. Dependency rules (by convention)

| Module | May import | Must not import |
|---|---|---|
| `core/` | `utils/`, `core/types.py` | Everything else under `src/` |
| `activations/`, `losses/`, `optimizers/` | `core/` | `models/`, `training/`, `data/`, `analysis/` |
| `layers/` | `core/` | `training/`, `data/`, `analysis/` |
| `models/` | `core/`, `layers/`, `activations/` | `training/`, `data/`, `analysis/`, `persistence/` |
| `data/` | `core/types.py`, `utils/` | `models/`, `training/`, `analysis/` |
| `metrics/` | `core/types.py` (and siblings in `metrics/`) | Other `src/` packages except as needed for metric math |
| `training/` | `core/`, `models/`, `losses/`, `optimizers/`, `metrics/`, `data/`, `utils/` | `analysis/` |
| `persistence/` | `core/`, `models/`, `config` | `training/`, `analysis/` |
| `config.py` | stdlib only | Rest of `src/` |
| `factory.py` | Almost all of `src/` | `analysis/` |
| Project `utils/` (`experiment_cli.py`, `run_parser.py`) | stdlib, `pathlib` | `src/` |
| `scripts/run_experiment.py` | `utils/`, later `src/` when training is wired | `analysis/` (unless explicitly orchestrating) |
| `run.py` | `utils/`, `scripts/` | — |
| `analysis/` | `pandas`, `matplotlib`, `numpy` (and optionally `persistence/` if you add thin adapters — keep off `training/` / `models/`) | **`training/`**, **`models/`** |

---

## 8. Config → objects

```
experiments_configs/xxx.json
      │  json.load()
      ▼
dict
      │  load_config()       (src/config.py)
      ▼
ExperimentConfig
      │  build_experiment()  (src/factory.py)
      ▼
(model, trainer, split_datasets)
```

`factory.py` exposes registry dicts: `ACTIVATIONS`, `LOSSES`, `OPTIMIZERS`, `METRICS`, `TRAINERS`, `INITIALIZERS` (string names for weight init), `STOPPING_CRITERIA` (stopping classes). Adding a new optimiser: implement class → register in `OPTIMIZERS` → use the string key in JSON.

---

## 9. Training loop (summary)

`BaseTrainer.fit`:
- Until `stopping.should_stop(history)`:
  - `train_epoch` (differs for online / batch / minibatch)
  - append losses, times, optional val loss, log epoch

All trainers share the same `forward → backward → optimizer.update` sequence; only batching differs.

---

## 10. Backpropagation data flow (summary)

Input flows forward through layers to ŷ. `loss.gradient(y, ŷ)` seeds backward. Each layer returns ∂L/∂input to the previous layer. Gradients are collected and passed to `optimizer.update`.

---

## 11. Experiment output

Each run writes to `experiments/<run_id>/`:

| File | Role |
|---|---|
| `config.json` | Snapshot of `ExperimentConfig` |
| `model.npz` | Weights |
| `history.json` | Per-epoch losses and times |
| `metrics.json` | Final evaluation metrics |

`run_id` is typically a timestamp; may be overridden by CLI.

---

## 12. Random state

Use `src.utils.random_state.make_rng(cfg.seed)` for all stochastic operations. Do not use global `np.random` APIs.

---

## 13. Extending the project

Same pattern as before: new class under the right package → export if needed → register in `factory.py` → reference by string in JSON.

---

## 14. Testing

Tests mirror modules (`tests/test_activations.py` ↔ `src/activations/`). Implement components in dependency order (activations → losses → optimizers → data → metrics → models → training).

---

## 15. Conventions

- Python ≥ 3.11, type hints on public APIs, Google- or NumPy-style docstrings (stay consistent).
- NumPy for numerics; pandas only where useful for CSV. No scikit-learn / torch / keras for core training.
- Prefer `Logger` from `src/utils/logger.py` over `print` inside training code; the runner may print a final line with the run path.
