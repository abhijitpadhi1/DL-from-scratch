import numpy as np

# Input "layer"
class Input:

    # Forward pass
    def forward(self, inputs: np.ndarray, training: bool) -> None:
        """
        Perform the forward pass of the Input layer.

        Parameters:
        inputs (np.ndarray): Input data.
        training (bool): Flag indicating whether the layer is in training mode.
        """
        self.output = inputs