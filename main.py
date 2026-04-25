from utils.parser import parse_args
from src.experiments import ejercicio_1, ejercicio_2, ejercicio_3

EJERCICIOS = {
    1: ejercicio_1.run,
    2: ejercicio_2.run,
    3: ejercicio_3.run,
}


def main():
    ejercicio, cfg = parse_args()
    EJERCICIOS[ejercicio](cfg)


if __name__ == "__main__":
    main()
