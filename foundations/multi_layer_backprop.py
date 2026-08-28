import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        # Forward pass
        # We traspose the weights to be able to multply due to the shape the wieghts have, see example below
        # z = x · Wᵀ + b

        #           ┌ 0.1  0.4 ┐
        # [1,2,3] · │ 0.2  0.5 │ + [0.1, 0.2]
        #           └ 0.3  0.6 ┘

        #    Wᵀ forma: (3, 2)

        #   z₁ = 1·0.1 + 2·0.2 + 3·0.3 = 1.4  + 0.1 (bias) = 1.5
        #   z₂ = 1·0.4 + 2·0.5 + 3·0.6 = 3.2  + 0.2 (bias) = 3.4

        #   z = [1.5, 3.4]    shape (2,)
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)
        
        
        z1 = x @ W1.T + b1
        a1 = np.maximum(0, z1) #shape (2,)
        z2 = a1 @ W2.T + b2 # shape (2,)
        loss = np.mean((z2 - y_true) **2) #scalar

        # Backward pass
        n = len(y_true) if y_true.ndim > 0 else 1
        dz2 = 2 * (z2 - y_true) / n # Loss function derivative, the  LF = mean((predictions - y_true)^2), shape (n,)
        db2 = dz2 # when we derivate with chain rule dL/dB  we end up with dZ2 * dZ2/dB2 = 1 then db2 = dz2 shape (1,)
        dW2 = dz2.reshape(-1,1) @ a1.reshape(1,-1)  # dL/dW2 by ChR is equal to dZ2 * activation, shape (1,2)
        # reshape (-1,n) I want  n columns, calculate the rows
        # reshape (n,-1) I want  n rows, calculate the columns
        # an (n,) array in numpy is not row nor column just a 1D without orientation
        # a1 shape = (1,2) for the weighted sum of the 2 inputs
        # so we reshape dz2 and a1 to be able to multiply
        da1 = dz2.reshape (1,-1) @ W2 # dL/dA2 by ChR is equal to dZ1 * weights, shape (1,2)
        # but as we are going backward we have to take care, that kind of shape do I need for the next step?
        # 
        da1 = da1.flatten() # shape (2,) same as a1 shape
        dz1 = da1 * (z1 > 0).astype(float) #The local derivative for a ReLu depends of the sign, thats why we use z1 > 0
        db1 = dz1
        dW1 = dz1.reshape(-1,1) @ x.reshape(1,-1)

        return {
            'loss': round(float(loss),4),
            'dW1': np.round(dW1,4).tolist(),
            'db1': np.round(db1,4).tolist(),
            'dW2': np.round(dW2,4).tolist(),
            'db2': np.round(db2,4).tolist(),
        }



