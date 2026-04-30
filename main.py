from tutorial.ejercicios import ejercicio_1, ejercicio_2, ejercicio_3
from utils.parser import parse_args
#from src.experiments import ejercicio_2, ejercicio_3, ejercicio_1
from src.experiments import tutorial_step, tutorial_linear, tutorial_non_linear, tutorial_mlp_tanh

EJERCICIOS = {
    1: ejercicio_1.run,
    2: ejercicio_2.run,
    3: ejercicio_3.run,
}

TUTORIALES = {
    "step":      tutorial_step.run,
    "linear":    tutorial_linear.run,
    "nonlinear": tutorial_non_linear.run,
    "mlp":       tutorial_mlp_tanh.run,
}


def main():
    mode, key, cfg = parse_args()
    if mode == "tutorial":
        TUTORIALES[key]()
    else:
        EJERCICIOS[key](cfg)


if __name__ == "__main__":
    main()
