import numpy as np

# Dense layer
class Dense:

    # Layer initialization
    def __init__(self, n_inputs: int, n_neurons: int,
                 weight_regularizer_l1: float = 0, weight_regularizer_l2: float = 0,
                 bias_regularizer_l1: float = 0, bias_regularizer_l2: float = 0) -> None:
        """
        Initialize the Dense layer with weights, biases, and regularization parameters.
        
        Parameters:
        n_inputs (int): Number of input features.
        n_neurons (int): Number of neurons in the layer.
        weight_regularizer_l1 (float): L1 regularization strength for weights.
        weight_regularizer_l2 (float): L2 regularization strength for weights.
        bias_regularizer_l1 (float): L1 regularization strength for biases.
        bias_regularizer_l2 (float): L2 regularization strength for biases.
        """
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    # Forward pass
    def forward(self, inputs: np.ndarray, training: bool) -> None:
        """
        Perform the forward pass of the Dense layer.
        
        Parameters:
        inputs (ndarray): Input data.
        training (bool): Flag indicating whether the layer is in training mode.
        """
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues: np.ndarray) -> None:
        """
        Perform the backward pass of the Dense layer.
        
        Parameters:
        dvalues (np.ndarray): Gradient of the loss with respect to the output of the layer.
        """
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    # Retrieve layer parameters
    def get_parameters(self) -> tuple:
        """
        Retrieve the weights and biases of the layer.
        
        Returns:
        tuple: A tuple containing the weights and biases.
        """
        return self.weights, self.biases

    # Set weights and biases in a layer instance
    def set_parameters(self, weights, biases) -> None:
        """
        Set the weights and biases of the layer.
        
        Parameters:
        weights (ndarray): Weights to set.
        biases (ndarray): Biases to set.
        """
        self.weights = weights
        self.biases = biases

