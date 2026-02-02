import numpy as np

# Linear activation
class Linear:

    # Forward pass
    def forward(self, inputs: np.ndarray, training: bool) -> None:
        """
        Perform the forward pass of the Linear activation function.

        Parameters:
        inputs (np.ndarray): Input data.
        training (bool): Flag indicating whether the layer is in training mode.
        """
        # Just remember values
        self.inputs = inputs
        self.output = inputs

    # Backward pass
    def backward(self, dvalues: np.ndarray) -> None:
        """
        Perform the backward pass of the Linear activation function.

        Parameters:
        dvalues (np.ndarray): Gradient of the loss with respect to the output of the layer.
        """
        # derivative is 1, 1 * dvalues = dvalues - the chain rule
        self.dinputs = dvalues.copy()

    # Calculate predictions for outputs
    def predictions(self, outputs: np.ndarray) -> np.ndarray:
        """
        Return the predictions for the given outputs.
        
        Parameters:
        outputs (np.ndarray): Output data from the forward pass.
        
        Returns:
        np.ndarray: Predictions based on the outputs.
        """
        return outputs