from src.config import ExperimentConfig
from src.network.multilayer_perceptron import MultilayerPerceptron


def save_run(
    run_id: str,
    cfg: ExperimentConfig,
    model: MultilayerPerceptron,
    history: dict,
    metrics: dict[str, float],
) -> None:
    """Guarda en results/<run_id>/: config.json, weights.npz, history.json, metrics.json"""
    raise NotImplementedError("TODO")


def load_run(run_id: str) -> dict:
    """Carga todo lo guardado por save_run.

    Returns:
        dict con claves: config, weights, history, metrics
    """
    raise NotImplementedError("TODO")
