from utils.parser import parse_args
from src.experiments import ejercicio_1, ejercicio_2, ejercicio_3
from src.experiments import tutorial_step, tutorial_linear, tutorial_non_linear

EJERCICIOS = {
    1: ejercicio_1.run,
    2: ejercicio_2.run,
    3: ejercicio_3.run,
}

TUTORIALES = {
    "step":      tutorial_step.run,
    "linear":    tutorial_linear.run,
    "nonlinear": tutorial_non_linear.run,
}


def main():
    mode, key, cfg = parse_args()
    if mode == "tutorial":
        TUTORIALES[key]()
    else:
        EJERCICIOS[key](cfg)


if __name__ == "__main__":
    main()
