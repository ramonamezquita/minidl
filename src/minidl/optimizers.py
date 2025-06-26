class SGD:

    def __init__(self, learning_rate: float = 1e-3):
        self.learning_rate = learning_rate

    def update_parameter(self, parameter: Parameter, learning_rate: float | None = None) -> None:
        if learning_rate is None:
            learning_rate = self.learning_rate

        new_values = parameter.values - learning_rate * parameter.gradient
        parameter.set(values)


def train(model, x, y, epochs = 1):
    
    criterion = MSELoss()
    optimizer = SGD()
    
    for i in range(epochs):
        output   = model.forward(x)
        loss     = criterion(output, y)
        gradient = criterion.gradient(output)

        # Set gradients.
        model.backward(gradient)

        for parameter in model.parameters():
            optimizer.update_parameter(parameter)

    return model