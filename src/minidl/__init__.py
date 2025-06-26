from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np


class ArrayInitializer(Protocol):
    def initialize_array(): ...


class Uniform:
    def __init__(self, *shape):
        self.shape = shape

    def initialize_array(self) -> np.ndarray:
        return np.random.rand(*self.shape)


class Parameter:
    def __init__(self, *shape, intializer: ArrayInitializer = Uniform()):
        self._values = intializer.initialize_array()
        self._gradient = None

    def set_gradient(self, gradient: np.ndarray) -> None:
        self._gradient = gradient

    def set(self, values: np.ndarray) -> None:
        self._values = values


class Layer(ABC):
    def __init__(self):
        self._parameters: dict[str, Parameter] = {}

    @abstractmethod
    def forward(self, x) -> np.ndarray:
        pass

    def __call__(self, x: np.ndarray):
        return self.forward(x)

    def register_parameter(self, name: str, value: np.ndarray) -> None:
        self._parameters[name] = value

    def parameters(self) -> dict:
        return self._parameters


class Linear(Layer):
    def __init__(self, features_in: int, features_out: int):
        super().__init__()
        self.W = Parameter(features_in, features_out)
        self.b = Parameter(1, features_out)

        self.register_parameter("W", self.W)
        self.register_parameter("b", self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Applies affine transformation to `x`.

        Parameters
        ----------
        x : ndarray, shape=(n_samples, features_in)
            Input array.

        Returns
        -------
        output: shape=(n_samples, features_out)
        """
        self._J_x = self.W
        self._J_w = x
        return x @ self.W + self.b

    def backward(self, gradient: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Computes the gradient of the function being differentiated wrt `W` and `b`.

        Parameters
        ----------
        gradient : ndarray, shape=(n_samples, features_out)
            The upstream gradient. That is, the gradient of the function being
            differentiated wrt to the `forward` output.
        """
        dx = self._J_x.T @ gradient
        dW = self._J_w.T @ gradient
        db = np.sum(gradient, axis=0, keepdims=True)

        self.W.set_gradient(dW)
        self.b.set_gradient(db)
        return dx


class Tanh:
    def forward(self, x: np.ndarray) -> np.ndarray:
        output = np.tanh(x)
        self._J = 1 - output**2
        return output

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        # Element-wise activation functions yield diagonal Jacobians.
        return self._J * gradient


class Sequential:

    def __init__(self, layers: None | list = None):
        self.layers = layers
        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        self._layers.append(layer)

    def backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)


class MSELoss:
    def forward(self, input: np.ndarray, target: np.ndarray) -> float:
        self._target = target
        return (1 / 2) * np.mean((target - input) ** 2)

    def __call__(self, input: np.ndarray, target: np.ndarray) -> float:
        return self.forward(input, target)

    def gradient(self, input: np.ndarray) -> np.ndarray:
        return np.mean(self._target - input)
