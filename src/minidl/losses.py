import numpy as np


class SSELoss:
    def __call__(self, input: np.ndarray, target: np.ndarray) -> float:
        self._input = input
        self._target = target
        return (1 / 2) * np.sum((input - target) ** 2)

    def gradient(self, input: np.ndarray | None = None) -> np.ndarray:
        if input is None:
            input = self._input
        return input - self._target
