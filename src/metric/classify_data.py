import numpy as np
from src.activation.activation import Array


def classify_data (zeta: Array, output: Array)->Array:


    if zeta.shape != output.shape:
        raise ValueError("zeta and output have different shapes")

    arr_size = zeta.size
    [false_pos, false_neg, true_pos, true_neg] = np.zeros(arr_size)
    for index in range(arr_size):
        # TODO: add this as parameter. We assume that prob. >= 0.8 is a positive classification
        adjusted_output = 1 if output[index] >= 0.8 else 0
        if zeta[index] == 1 and adjusted_output == 1:
            true_pos += 1
        elif zeta[index] == 0 and adjusted_output == 1:
            false_pos += 1
        elif zeta[index] == 1 and adjusted_output == 0:
            false_neg += 1
        elif zeta[index] == 0 and adjusted_output == 0:
            true_neg += 1

    return np.array([false_pos, false_neg, true_pos, true_neg])
