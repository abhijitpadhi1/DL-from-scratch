import numpy as np

# RMSprop optimizer
class RMSprop:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate: float = 0.001, decay: float = 0., 
                    epsilon: float = 1e-7, rho: float = 0.9) -> None:
        """
        Initialize the RMSprop optimizer with given parameters.

        Parameters:
        learning_rate (float): Initial learning rate.
        decay (float): Learning rate decay over iterations.
        epsilon (float): Small constant to avoid division by zero.
        rho (float): Decay rate for the moving average of squared gradients.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # Call once before any parameter updates
    def pre_update_params(self) -> None:
        """
        Apply decay to learning rate
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer) -> None:
        """
        Update parameters of the given layer using RMSprop optimization.

        Parameters:
        layer (object): The layer to update, which must have weights, biases, dweights, and dbiases attributes.
        """

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self) -> None:
        """
        Increment the iteration count after parameter updates.
        """
        self.iterations += 1
