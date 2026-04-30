import argparse
from src.config import ExperimentConfig, load_config
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Construye el ArgumentParser con todos los flags del proyecto."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="SIA TP3 — Redes Neuronales",
    )

    # --ejercicio y --tutorial son mutuamente excluyentes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--ejercicio", "-e",
        type=int,
        choices=[1, 2, 3],
        metavar="{1,2,3}",
        help="Número de ejercicio a correr (requiere --config)",
    )
    group.add_argument(
        "--tutorial", "-t",
        choices=["step", "linear", "nonlinear", "mlp"],
        metavar="{step,linear,nonlinear,mlp}",
        help="Tutorial de sanidad a correr (no requiere --config)",
    )

    # --config: obligatorio para ejercicios, ignorado para tutoriales
    parser.add_argument(
        "--config", "-c",
        type=Path,
        metavar="PATH",
        help="Path al JSON de configuración (obligatorio con --ejercicio)",
    )

    # Overrides opcionales — sobreescriben cualquier valor del JSON
    parser.add_argument("--eta",           type=float, help="Tasa de aprendizaje η")
    parser.add_argument("--epochs",        type=int,   help="Máximo de épocas")
    parser.add_argument("--epsilon",       type=float, help="Umbral de convergencia ε")
    parser.add_argument("--activation",    type=str,   choices=["step", "identity", "tanh", "logistic", "relu"])
    parser.add_argument("--beta",          type=float, help="Parámetro β para tanh/logistic")
    parser.add_argument("--optimizer",     type=str,   choices=["gradient_descent", "momentum", "adam"])
    parser.add_argument("--training-mode", type=str,   choices=["online", "batch", "minibatch"], dest="training_mode")
    parser.add_argument("--batch-size",    type=int,   dest="batch_size")
    parser.add_argument("--seed",          type=int)
    parser.add_argument("--momentum-beta", type=float, dest="momentum_beta")
    parser.add_argument("--adam-beta1",    type=float, dest="adam_beta1")
    parser.add_argument("--adam-beta2",    type=float, dest="adam_beta2")

    return parser


def parse_args() -> tuple[str, int | str, ExperimentConfig | None]:
    """Parsea sys.argv. Carga el JSON base y sobreescribe con los flags CLI explícitos.

    Returns:
        ("ejercicio", número: int, cfg: ExperimentConfig)
        ("tutorial",  nombre: str, None)
    """
    parser = build_parser()
    args = parser.parse_args()

    # ── TUTORIAL — no necesita config ────────────────────────────────────
    if args.tutorial is not None:
        return ("tutorial", args.tutorial, None)

    # ── EJERCICIO — requiere --config ─────────────────────────────────────
    if args.config is None:
        parser.error("--ejercicio requiere --config PATH")

    cfg = load_config(args.config)

    # Sobreescribir con los flags CLI que se hayan pasado explícitamente
    overrides = {
        "eta":           args.eta,
        "epochs":        args.epochs,
        "epsilon":       args.epsilon,
        "activation":    args.activation,
        "beta":          args.beta,
        "optimizer":     args.optimizer,
        "training_mode": args.training_mode,
        "batch_size":    args.batch_size,
        "seed":          args.seed,
        "momentum_beta": args.momentum_beta,
        "adam_beta1":    args.adam_beta1,
        "adam_beta2":    args.adam_beta2,
    }
    for field, value in overrides.items():
        if value is not None:
            setattr(cfg, field, value)

    return "ejercicio", args.ejercicio, cfg
