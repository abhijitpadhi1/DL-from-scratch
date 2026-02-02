import numpy as np

# Sigmoid activation
class Sigmoid:

    # Forward pass
    def forward(self, inputs: np.ndarray, training: bool) -> None:
        """
        Perform the forward pass of the Sigmoid activation function.

        Parameters:
        inputs (np.ndarray): Input data.
        training (bool): Flag indicating whether the layer is in training mode.
        """
        # Save input and calculate/save output
        # of the sigmoid function
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    # Backward pass
    def backward(self, dvalues: np.ndarray) -> None:
        """
        Perform the backward pass of the Sigmoid activation function.

        Parameters:
        dvalues (np.ndarray): Gradient of the loss with respect to the output of the layer.
        """
        # Derivative - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output

    # Calculate predictions for outputs
    def predictions(self, outputs: np.ndarray) -> np.ndarray:
        """
        Return the predictions for the given outputs.
        
        Parameters:
        outputs (np.ndarray): Output data from the forward pass.
        
        Returns:
        np.ndarray: Predictions based on the outputs.
        """
        return (outputs > 0.5) * 1
