import numpy as np

# Softmax activation
class Softmax:

    # Forward pass
    def forward(self, inputs: np.ndarray, training: bool) -> None:
        """
        Perform the forward pass of the Softmax activation function.

        Parameters:
        inputs (np.ndarray): Input data.
        training (bool): Flag indicating whether the layer is in training mode.
        """
        # Remember input values
        self.inputs = inputs

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    # Backward pass
    def backward(self, dvalues: np.ndarray) -> None:
        """
        Perform the backward pass of the Softmax activation function.

        Parameters:
        dvalues (np.ndarray): Gradient of the loss with respect to the output of the layer.
        """

        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    # Calculate predictions for outputs
    def predictions(self, outputs: np.ndarray) -> np.ndarray:
        """
        Return the predictions for the given outputs.
        
        Parameters:
        outputs (np.ndarray): Output data from the forward pass.
        
        Returns:
        np.ndarray: Predictions based on the outputs.
        """
        return np.argmax(outputs, axis=1)
