import numpy as np
from src.activation.activation import Array


def classify_data (zeta: Array, output: Array)->Array:

    zeta = np.asarray(zeta).reshape(-1)
    output = np.asarray(output).reshape(-1)

    arr_output_size = output.shape[0]
    arr_zeta_size = zeta.shape[0]
    if arr_output_size != arr_zeta_size:
        raise ValueError("zeta and output have different shapes")

    false_pos, false_neg, true_pos, true_neg = np.zeros(4)
    for index in range(arr_output_size):
        # TODO: add this as parameter. We assume that prob. >= 0.8 is a positive classification
        adjusted_output = 1 if output[index] >= 0.8 else 0
        print(f"Index: {index}  Output: {output[index]:.4f}  Adjusted Output: {adjusted_output}  Zeta: {zeta[index]}")
        # TODO: aca no estoy usando flagged_fraud sino big_model_fraud_probability. Lo cual esta mal.
        if zeta[index] == 1 and adjusted_output == 1:
            true_pos += 1
        elif zeta[index] == 0 and adjusted_output == 1:
            false_pos += 1
        elif zeta[index] == 1 and adjusted_output == 0:
            false_neg += 1
        elif zeta[index] == 0 and adjusted_output == 0:
            true_neg += 1

    return np.array([false_pos, false_neg, true_pos, true_neg])
