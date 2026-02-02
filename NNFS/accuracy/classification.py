import numpy as np

from accuracy.accuracy import Accuracy

# Accuracy calculation for classification model
class Categorical(Accuracy):

    def __init__(self, *, binary: bool = False) -> None:
        """
        Initializes the Accuracy_Categorical class.
        
        Parameters:
        binary (bool): Indicates if the accuracy calculation is for binary classification.
        """
        # Binary mode?
        self.binary = binary

    # No initialization is needed
    def init(self, y) -> None:
        pass

    # Compares predictions to the ground truth values
    def compare(self, predictions: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compares predictions to the ground truth values.

        Parameters:
        predictions (np.ndarray): The predicted values.
        y (np.ndarray): The ground truth values.

        Returns:
        np.ndarray: Boolean array indicating correct predictions.
        """
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y