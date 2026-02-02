import numpy as np
from abc import ABC, abstractmethod

# Common accuracy class
class Accuracy(ABC):

    # Compares predictions to the ground truth values
    @abstractmethod
    def compare(self, predictions: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compares predictions to the ground truth values.

        Parameters:
        predictions (np.ndarray): The predicted values.
        y (np.ndarray): The ground truth values.

        Returns:
        np.ndarray: Boolean array indicating correct predictions.
        """
        pass

    # Calculates an accuracy
    # given predictions and ground truth values
    def calculate(self, predictions: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates an accuracy given predictions and ground truth values.

        Parameters:
        predictions (np.ndarray): The predicted values.
        y (np.ndarray): The ground truth values.

        Returns:
        float: The calculated accuracy.
        """
        # Get comparison results
        comparisons = self.compare(predictions, y)

        # Calculate an accuracy
        accuracy = np.mean(comparisons)

        # Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        # Return accuracy
        return float(accuracy)

    # Calculates accumulated accuracy
    def calculate_accumulated(self) -> float:
        """
        Calculates accumulated accuracy.

        Returns:
        float: The accumulated accuracy.
        """
        # Calculate an accuracy
        accuracy = self.accumulated_sum / self.accumulated_count

        # Return the data and regularization losses
        return accuracy

    # Reset variables for accumulated accuracy
    def new_pass(self) -> None:
        """
        Resets variables for accumulated accuracy.
        """
        self.accumulated_sum = 0
        self.accumulated_count = 0
