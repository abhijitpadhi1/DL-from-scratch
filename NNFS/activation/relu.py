import numpy as np

# ReLU activation
class ReLU:

    # Forward pass
    def forward(self, inputs: np.ndarray, training: bool) -> None:
        """
        Perform the forward pass of the ReLU activation function.

        Parameters:
        inputs (np.ndarray): Input data.
        training (bool): Flag indicating whether the layer is in training mode.
        """
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues: np.ndarray) -> None:
        """
        Perform the backward pass of the ReLU activation function.

        Parameters:
        dvalues (np.ndarray): Gradient of the loss with respect to the output of the layer.
        """
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

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