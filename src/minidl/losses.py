import numpy as np


class ScalarFunction:
    pass

class Differentiable:
    pass


class MSELoss:
    def forward(self, input: np.ndarray, target: np.ndarray) -> float:
        self._input = input
        self._target = target
        return (1 / 2) * np.mean((target - input) ** 2)

    def backward(self, gradient: np.ndarray = None) -> np.ndarray:
        return np.mean(self._target - self._input)
