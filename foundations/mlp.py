import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        layer1 = x

        #
        #    1st layer = (x[1.0, 2.0] * w[0.1, 0.2], + b[0.1 0.1]), i=0
        #                               w[0.3, 0.4] + b[0.1 0.1])  
        #    2nd layer = (x[1.0, 2.0] * w[0.5], + b[0.0]), i=1
        #                               w[0.6] + b[0.0]) 
        #
        #
        #
        for i in range (len(weights)):
            layer1 = layer1 @ weights[i] + biases[i]
            if i < len(weights)-1:   #Relu is applied only for the 1st layer
                 layer1 = np.maximum(0, layer1)

        return np.round(layer1,4)