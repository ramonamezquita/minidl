import numpy as np


class Tanh:
    def forward(self, x: np.ndarray) -> np.ndarray:
        output = np.tanh(x)
        self._J = 1 - output**2
        return output

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        # Element-wise activation functions yield diagonal Jacobians.
        return self._J * gradient