from typing import Protocol

import numpy as np


class ArrayInitializer(Protocol):
    def initialize_array(): ...


class RandomUniform:
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        self.min_val = min_val
        self.max_val = max_val

    def initialize_array(self, shape) -> np.ndarray:
        return np.random.rand(shape)
