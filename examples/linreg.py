import numpy as np

from minidl.layers import Linear
from minidl.losses import SSELoss
from minidl.optimizers import SGD


def main():

    # Generate dummy data.
    n_samples = 100
    features_in = 2
    features_out = 2
    true_weight = np.random.rand(features_in, features_out)
    true_bias = np.random.rand(features_out)
    x = np.random.randn(n_samples, features_in)
    y = x @ true_weight + true_bias

    # model + loss + optimizer.
    model = Linear(features_in, features_out)
    loss_fn = SSELoss()
    optimizer = SGD(learning_rate=0.001)
    n_iterations = 1000

    
    for j in range(n_iterations):
        output   = model.forward(x)
        loss     = loss_fn(output, y)
        gradient = loss_fn.gradient()

        # Set gradients.
        model.backward(gradient)

        if j % 25 == 0:
            print(f"[iteration {j}] loss: {loss}")

        for parameter in model.parameters:
            optimizer.update_parameter(parameter)
        
        j += 1

if __name__ == "__main__":
    main()
