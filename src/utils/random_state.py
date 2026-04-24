import numpy as np


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Create a NumPy random generator from a seed.

    All randomness in the project must flow through a generator produced
    by this function.  Never call ``np.random.xxx`` directly.

    Args:
        seed: Integer seed for reproducibility.  ``None`` uses a random seed.

    Returns:
        A ``numpy.random.Generator`` instance.
    """
    return np.random.default_rng(seed)
