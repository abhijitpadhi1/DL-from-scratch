import numpy as np

from loss.loss import Loss

# Mean Squared Error loss
class MeanSquaredError(Loss):  # L2 loss

    # Forward pass
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Calculates the mean squared error loss for each sample.

        Parameters:
        y_pred (np.ndarray): Predicted values.
        y_true (np.ndarray): True values.

        Returns:
        np.ndarray: Loss for each sample.
        """
        # Calculate loss
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)

        # Return losses
        return sample_losses

    # Backward pass
    def backward(self, dvalues: np.ndarray, y_true: np.ndarray) -> None:
        """
        Calculates the gradient of the mean squared error loss.

        Parameters:
        dvalues (np.ndarray): Predicted values.
        y_true (np.ndarray): True values.
        """
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Mean Absolute Error loss
class MeanAbsoluteError(Loss):  # L1 loss

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Calculates the mean absolute error loss for each sample.

        Parameters:
        y_pred (np.ndarray): Predicted values.
        y_true (np.ndarray): True values.

        Returns:
        np.ndarray: Loss for each sample.
        """
        # Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

        # Return losses
        return sample_losses


    # Backward pass
    def backward(self, dvalues: np.ndarray, y_true: np.ndarray) -> None:
        """
        Calculates the gradient of the mean absolute error loss.

        Parameters:
        dvalues (np.ndarray): Predicted values.
        y_true (np.ndarray): True values.
        """
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Calculate gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples
