from abc import ABC, abstractmethod
import numpy as np

# Common loss class
class Loss(ABC):

    # Forward pass
    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Calculates the loss for each sample.

        Parameters:
        y_pred (np.ndarray): Predicted values.
        y_true (np.ndarray): True values.

        Returns:
        np.ndarray: Loss for each sample.
        """
        pass

    # Regularization loss calculation
    def regularization_loss(self) -> float:
        """
        Calculates the regularization loss

        Returns:
        regularization_loss (float): The calculated regularization loss
        """
        # 0 by default
        regularization_loss = 0

        # Calculate regularization loss
        # iterate all trainable layers
        for layer in self.trainable_layers:

            # L1 regularization - weights
            # calculate only when factor greater than 0
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

            # L2 regularization - weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

            # L1 regularization - biases
            # calculate only when factor greater than 0
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

            # L2 regularization - biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss

    # Set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers) -> None:
        """
        Sets the trainable layers for the loss calculation.

        Parameters:
        trainable_layers (list): List of trainable layers.
        """
        self.trainable_layers = trainable_layers


    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output: np.ndarray, y: np.ndarray, *, 
                    include_regularization: bool = False) -> tuple[float, float] | float:
        """
        Calculates the data and regularization losses

        Parameters:
        output (np.ndarray): The model output.
        y (np.ndarray): The ground truth values.
        include_regularization (bool): Whether to include regularization loss.

        Returns:
        float or tuple: The calculated loss. If include_regularization is True,
                        returns a tuple of (data_loss, regularization_loss).
        """

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        # If just data loss - return it
        if not include_regularization:
            return float(data_loss)

        # Return the data and regularization losses
        return float(data_loss), float(self.regularization_loss())

    # Calculates accumulated loss
    def calculate_accumulated(self, *, 
                              include_regularization: bool = False) -> float | tuple[float, float]:
        """
        Calculates the accumulated data and regularization losses

        Parameters:
        include_regularization (bool): Whether to include regularization loss.

        Returns:
        float or tuple: The calculated loss. If include_regularization is True,
                        returns a tuple of (data_loss, regularization_loss).
        """

        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count

        # If just data loss - return it
        if not include_regularization:
            return data_loss

        # Return the data and regularization losses
        return data_loss, self.regularization_loss()

    # Reset variables for accumulated loss
    def new_pass(self) -> None:
        """
        Resets the accumulated sum and count for loss calculation.
        """
        self.accumulated_sum = 0
        self.accumulated_count = 0
