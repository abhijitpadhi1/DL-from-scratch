import numpy as np

from loss.loss import Loss

# Cross-entropy loss
class CategoricalCrossentropy(Loss):
    
    # Forward pass
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Calculates the categorical cross-entropy loss for each sample.

        Parameters:
        y_pred (np.ndarray): Predicted probabilities.
        y_true (np.ndarray): True labels.

        Returns:
        np.ndarray: Loss for each sample.
        """
        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        else:
            raise ValueError("y_true must be either 1D or 2D array")

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues: np.ndarray, y_true: np.ndarray) -> None:
        """
        Calculates the gradient of the categorical cross-entropy loss.

        Parameters:
        dvalues (np.ndarray): Predicted probabilities.
        y_true (np.ndarray): True labels.
        """
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples



# Binary cross-entropy loss
class BinaryCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Calculates the binary cross-entropy loss for each sample.

        Parameters:
        y_pred (np.ndarray): Predicted probabilities.
        y_true (np.ndarray): True labels.

        Returns:
        np.ndarray: Loss for each sample.
        """
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        # Return losses
        return sample_losses

    # Backward pass
    def backward(self, dvalues: np.ndarray, y_true: np.ndarray) -> None:
        """
        Calculates the gradient of the binary cross-entropy loss.

        Parameters:
        dvalues (np.ndarray): Predicted probabilities.
        y_true (np.ndarray): True labels.
        """
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples
