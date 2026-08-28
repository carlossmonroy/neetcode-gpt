import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        #We use matmul to do the dotproduct to do the multiplication we want
        #Dot product (X x W):
        #X =[[1, 2, 3],   weights = [0.1, 0.2, 0.3]
        # [4, 5, 6],
        # [7, 8, 9]]
        #prediction[0] = 1*0.1 + 2*0.2 + 3*0.3 = 1.4
        #prediction[1] = 4*0.1 + 5*0.2 + 6*0.3 = 3.2
        #prediction[2] = 7*0.1 + 8*0.2 + 9*0.3 = 5.0
        return np.round(np.matmul(X,weights),5) 

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        MSE = np.mean (np.square(model_prediction - ground_truth))
        return np.round(MSE,5)
