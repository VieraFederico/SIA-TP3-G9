from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Array: TypeAlias = NDArray[np.float64]
Shape: TypeAlias = tuple[int, ...]
