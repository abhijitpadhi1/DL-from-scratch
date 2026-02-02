import numpy as np

from accuracy.accuracy import Accuracy

# Accuracy calculation for regression model
class Regression(Accuracy):

    def __init__(self) -> None:
        """
        Initializes the Accuracy_Regression class.
        """
        # Create precision property
        self.precision = None

    # Calculates precision value
    # based on passed-in ground truth values
    def init(self, y: np.ndarray, reinit: bool = False) -> None:
        """
        Calculates precision value based on passed-in ground truth values.

        Parameters:
        y (np.ndarray): The ground truth values.
        reinit (bool): Whether to reinitialize the precision value.
        """
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

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
        return np.absolute(predictions - y) < self.precision
