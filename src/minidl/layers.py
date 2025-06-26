import numpy as np

from .initializers import ArrayInitializer, RandomUniform


class Parameter:
    def __init__(self, *shape, intializer: ArrayInitializer = RandomUniform()):
        self._values = intializer.initialize_array(shape)
        self._gradient = None

    def set_gradient(self, gradient: np.ndarray) -> None:
        self._gradient = gradient

    def set(self, values: np.ndarray) -> None:
        self._values = values


class Layer:
    def __init__(self):
        self._parameters: dict[str, Parameter] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Performs the forward pass of the layer and returns the output.

        Parameters
        ----------
        x : np.ndarray
            The input to the layer.

        Returns
        -------
        np.ndarray
            The output of the layer after applying its transformation to the input.
        """
        pass

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """Returns the gradient of the function being differentiated wrt the input of the layer.

        The function being differentiated is typically a loss, e.g., the MSE.

        Parameters
        ----------
        gradient : ndarray
            The gradient of the function being differentiated wrt the output of this layer
            (i.e., the upstream gradient).

        Returns
        -------
        np.ndarray
            The gradient of the function being differentiated wrt the input
        """
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
        """Backward pass.

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


class Sequential:

    def __init__(self, layers: None | list = None):
        self._layers = layers
        if layers:
            for layer in layers:
                self.add(layer)
        else:
            self._layers = []

    def add(self, layer):
        self._layers.append(layer)

    def backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
