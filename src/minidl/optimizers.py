from .layers import Parameter


class SGD:

    def __init__(self, learning_rate: float = 1e-3):
        self.learning_rate = learning_rate

    def update_parameter(self, parameter: Parameter) -> None:
        updated_value = parameter.value - self.learning_rate * parameter.gradient
        parameter.set(updated_value)
