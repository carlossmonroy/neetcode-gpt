import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        # dL/dW -> Derivative of loss with respect to the weight, how is the weight connected to the Loss
        # sigmoid = y_hat
        # Chain Rule --> dL/dW = dL/dy_hat * d/z * dz/dW
        # dL/dy_hat = y_hat - y (error)
        # dy_hat/dz = y_hat(1-y_hat)
        # dsigmoid/dw = x

        z = x @ w + b
        y_hat = 1/ (1 + np.exp(-z))

        dL_dy_hat = y_hat - y_true
        error = dL_dy_hat

        dy_hat_dZ = y_hat * (1 - y_hat)

        dZ_dW = x

        dL_dw = np.round( error * y_hat * (1-y_hat) * x,5)
        dL_db = np.round(error * y_hat * (1-y_hat), 5)
        return (dL_dw, dL_db)
