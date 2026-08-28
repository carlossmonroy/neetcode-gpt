import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7
        np.clip(y_pred, epsilon, 1-epsilon) 
        L = -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred)) # Formula says - 1/n * Sum of... = MEAN
        return round(L,4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7
        np.clip(y_pred, epsilon, 1-epsilon) 
        L = -np.mean(np.sum(y_true * np.log(y_pred), axis=1)) # Axis 1 means sum all values in rows, 0 is columns
        return round(L,4)
