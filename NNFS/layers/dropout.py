import numpy as np

# Dropout Layer
class Dropout:

    # Init
    def __init__(self, rate: float) -> None:
        """
        Initialize the Dropout layer with a given dropout rate.
        
        Parameters:
        rate (float): Dropout rate, the fraction of inputs to drop.
        """
        # Store rate, we invert it as for example for dropout
        # of 0.1 we need success rate of 0.9
        self.rate = 1 - rate

    # Forward pass
    def forward(self, inputs: np.ndarray, training: bool) -> None:
        """
        Perform the forward pass of the Dropout layer.
        
        Parameters:
        inputs (np.ndarray): Input data.
        training (bool): Flag indicating whether the layer is in training mode.
        """
        # Save input values
        self.inputs = inputs


        # If not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            return

        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    # Backward pass
    def backward(self, dvalues: np.ndarray) -> None:
        """
        Perform the backward pass of the Dropout layer.
        
        Parameters:
        dvalues (np.ndarray): Gradient of the loss with respect to the output of the layer.
        """
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask



