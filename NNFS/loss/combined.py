import numpy as np

# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():

    # Backward pass
    def backward(self, dvalues: np.ndarray, y_true: np.ndarray) -> None:
        """
        Performs the backward pass for the combined Softmax activation
        and categorical cross-entropy loss.

        Parameters:
        dvalues (np.ndarray): Gradient of the loss with respect to the output.
        y_true (np.ndarray): True labels.
        """
        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples