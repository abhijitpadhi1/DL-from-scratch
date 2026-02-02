import numpy as np
from tqdm import tqdm
import pickle
import copy

from layers.input import Input
from activation.softmax import Softmax
from loss.classification import CategoricalCrossentropy
from loss.combined import Activation_Softmax_Loss_CategoricalCrossentropy

# Model class
class Model:
    def __init__(self) -> None:
        """
        Initializes the Model class.
        """
        # Create a list of network objects
        self.layers = []
        # Softmax classifier's output object
        self.softmax_classifier_output = None


    # Add objects to the model
    def add(self, layer: object) -> None:
        """
        Adds a layer or object to the model.

        Parameters:
        layer (object): The layer or object to add to the model.
        """
        self.layers.append(layer)


    # Set loss, optimizer and accuracy
    def set(self, *, loss=None, optimizer=None, accuracy=None) -> None:
        """
        Sets the loss, optimizer, and accuracy objects for the model.

        Parameters:
        loss (object): The loss object.
        optimizer (object): The optimizer object.
        accuracy (object): The accuracy object.
        """
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy


    # Finalize the model
    def finalize(self) -> None:
        """
        Finalizes the model by setting up the input layer,
        linking layers, and preparing for training.
        """
        # Create and set the input layer
        self.input_layer = Input()

        # Count all the objects
        layer_count = len(self.layers)

        # Initialize a list containing trainable layers:
        self.trainable_layers = []

        # Iterate the objects
        for i in range(layer_count):
            # If it's the first layer,
            # the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            # All layers except for the first and the last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            # The last layer - the next object is the loss
            # Also let's save aside the reference to the last object
            # whose output is the model's output
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # If layer contains an attribute called "weights",
            # it's a trainable layer -
            # add it to the list of trainable layers
            # We don't need to check for biases -
            # checking for weights is enough
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        # Update loss object with trainable layers
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        # If output activation is Softmax and
        # loss function is Categorical Cross-Entropy
        # create an object of combined activation
        # and loss function containing
        # faster gradient calculation
        if isinstance(self.layers[-1], Softmax) and isinstance(self.loss, CategoricalCrossentropy):
            # Create an object of combined activation
            # and loss functions
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()


    # Prints compact model summary (Dense layers only)
    def summary(self) -> None:
        """
        Prints a compact summary of the model, showing Dense layers only.
        """
        print("\n" + "-" * 65)
        print(f"{'Layer (type)':<25}{'Shape (in, out)':<25}{'Param #':<10}")
        print("=" * 65)

        total_params = 0
        trainable_params = 0

        for layer in self.layers:

            # Only show layers with weights (trainable layers)
            if not hasattr(layer, "weights"):
                continue

            layer_name = layer.__class__.__name__

            # Get input & output size
            input_size = layer.weights.shape[0]
            output_size = layer.weights.shape[1]

            shape = f"({input_size}, {output_size})"

            # Count parameters
            params = layer.weights.size + layer.biases.size

            total_params += params
            trainable_params += params

            print(f"{layer_name:<25}{shape:<25}{params:<10}")

        print("=" * 65)
        print(f"Total params: {total_params}")
        print(f"Trainable params: {trainable_params}")
        print(f"Non-trainable params: {total_params - trainable_params}")
        print("-" * 65 + "\n")


    # Train the model
    def train(self, X: np.ndarray, y: np.ndarray, *, epochs: int = 1, batch_size: int | None = None, 
                    print_every: int = 1, validation_data: tuple | None = None) -> None:
        """
        Trains the model using the provided data.

        Parameters:
        X (array-like): Input data.
        y (array-like): Target labels.
        epochs (int): Number of epochs to train.
        batch_size (int or None): Size of each training batch.
        print_every (int): Frequency of printing training progress.
        validation_data (tuple or None): Validation data (X_val, y_val).
        """
        # Initialize accuracy object
        self.accuracy.init(y)

        # Default value if batch size is not being set
        train_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            train_steps = len(X) // batch_size
            # Dividing rounds down. If there are some remaining
            # data but not a full batch, this won't include it
            # Add `1` to include this not full batch
            if train_steps * batch_size < len(X):
                train_steps += 1

        # Main training loop
        for epoch in range(1, epochs+1):

            # Print epoch number
            print(f'\nepoch: {epoch}/{epochs}')

            # Reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()

            # Iterate over steps
            for step in tqdm(range(train_steps)):
                # If batch size is not set -
                # train using one step and full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                # Otherwise slice a batch
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                # Perform the forward pass
                output = self.forward(batch_X, training=True)

                # Calculate loss
                data_loss, regularization_loss = self.loss.calculate(
                    output, batch_y, include_regularization=True
                )
                loss = data_loss + regularization_loss

                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                # Perform backward pass
                self.backward(output, batch_y)

                # Optimize (update parameters)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                # # Print a summary
                # if not step % print_every or step == train_steps - 1:
                #     print(f'step: {step}, ' +
                #           f'acc: {accuracy:.3f}, ' +
                #           f'loss: {loss:.3f} (' +
                #           f'data_loss: {data_loss:.3f}, ' +
                #           f'reg_loss: {regularization_loss:.3f}), ' +
                #           f'lr: {self.optimizer.current_learning_rate}')

            # Get and print epoch loss and accuracy
            epoch_data_loss, epoch_regularization_loss = \
                self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'training => ' +
                  f'acc: {epoch_accuracy:.3f} - ' +
                  f'loss: {epoch_loss:.3f} - (' +
                  f'data_loss: {epoch_data_loss:.3f} - ' +
                  f'reg_loss: {epoch_regularization_loss:.3f}) - ' +
                  f'lr: {self.optimizer.current_learning_rate}')

            # If there is the validation data
            if validation_data is not None:
                # Evaluate the model:
                self.evaluate(*validation_data, batch_size=batch_size)


    # Evaluates the model using passed-in dataset
    def evaluate(self, X_val: np.ndarray, y_val: np.ndarray, *, batch_size: int | None = None) -> None:
        """
        Evaluates the model using the provided validation data.

        Parameters:
        X_val (np.ndarray): Validation input data.
        y_val (np.ndarray): Validation target labels.
        batch_size (int or None): Size of each evaluation batch.
        """
        # Default value if batch size is not being set
        validation_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            # Dividing rounds down. If there are some remaining
            # data but not a full batch, this won't include it
            # Add `1` to include this not full batch
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        # Reset accumulated values in loss
        # and accuracy objects
        self.loss.new_pass()
        self.accuracy.new_pass()

        # Iterate over steps
        for step in range(validation_steps):
            # If batch size is not set -
            # train using one step and full dataset
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            # Otherwise slice a batch
            else:
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]

            # Perform the forward pass
            output = self.forward(batch_X, training=False)

            # Calculate the loss
            self.loss.calculate(output, batch_y)

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        # Get and print validation loss and accuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        # Print a summary
        print(f'validation => ' +
              f'acc: {validation_accuracy:.3f} - ' +
              f'loss: {validation_loss:.3f}')


    # Predicts on the samples
    def predict(self, X: np.ndarray, *, batch_size : int | None = None) -> np.ndarray:
        """
        Predicts output for the given input samples.

        Parameters:
        X (np.ndarray): Input data samples.
        batch_size (int or None): Size of each prediction batch.

        Returns:
        np.ndarray: Predicted outputs.
        """
        # Default value if batch size is not being set
        prediction_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            prediction_steps = len(X) // batch_size

            # Dividing rounds down. If there are some remaining
            # data but not a full batch, this won't include it
            # Add `1` to include this not full batch
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        # Model outputs
        output = []

        # Iterate over steps
        for step in range(prediction_steps):

            # If batch size is not set -
            # train using one step and full dataset
            if batch_size is None:
                batch_X = X
            # Otherwise slice a batch
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]

            # Perform the forward pass
            batch_output = self.forward(batch_X, training=False)

            # Append batch prediction to the list of predictions
            output.append(batch_output)

        # Stack and return results
        return np.vstack(output)


    # Performs forward pass
    def forward(self, X: np.ndarray, training: bool) -> np.ndarray:
        """
        Performs a forward pass through the network.

        Parameters:
        X (np.ndarray): Input data.
        training (bool): Whether the forward pass is during training.

        Returns:
        np.ndarray: Output of the network.
        """
        # Call forward method on the input layer
        # this will set the output property that
        # the first layer in "prev" object is expecting
        self.input_layer.forward(X, training)

        # Call forward method of every object in a chain
        # Pass output of the previous object as a parameter
        layer = None
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # "layer" is now the last object from the list,
        # return its output
        if layer is not None:
            return layer.output
        else:
            return self.input_layer.output


    # Performs backward pass
    def backward(self, output: np.ndarray, y: np.ndarray) -> None:
        """
        Performs a backward pass through the network.

        Parameters:
        output (np.ndarray): Output of the network.
        y (np.ndarray): True labels.
        """
        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # First call backward method
            # on the combined activation/loss
            # this will set dinputs property
            self.softmax_classifier_output.backward(output, y)

            # Since we'll not call backward method of the last layer
            # which is Softmax activation
            # as we used combined activation/loss
            # object, let's set dinputs in this object
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            # Call backward method going through
            # all the objects but last
            # in reversed order passing dinputs as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return

        # First call backward method on the loss
        # this will set dinputs property that the last
        # layer will try to access shortly
        self.loss.backward(output, y)

        # Call backward method going through all the objects
        # in reversed order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)


    # Retrieves and returns parameters of trainable layers
    def get_parameters(self) -> list:
        """
        Retrieves and returns parameters of trainable layers.

        Returns:
        list: A list of parameters from trainable layers.
        """
        # Create a list for parameters
        parameters = []

        # Iterable trainable layers and get their parameters
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())

        # Return a list
        return parameters


    # Updates the model with new parameters
    def set_parameters(self, parameters: list) -> None:
        """
        Updates the model with new parameters.

        Parameters:
        parameters (list): A list of parameters to update the model with.
        """
        # Iterate over the parameters and layers
        # and update each layers with each set of the parameters
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)


    # Saves the parameters to a file
    def save_parameters(self, path: str) -> None:
        """
        Saves the parameters to a file.

        Parameters:
        path (str): The file path to save the parameters.
        """
        # Open a file in the binary-write mode
        # and save parameters into it
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)


    # Loads the weights and updates a model instance with them
    def load_parameters(self, path: str) -> None:
        """
        Loads the weights and updates a model instance with them.

        Parameters:
        path (str): The file path to load the parameters from.
        """
        # Open file in the binary-read mode,
        # load weights and update trainable layers
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))


    # Saves the model
    def save(self, path: str) -> None:
        """
        Saves the model to a file.

        Parameters:
        path (str): The file path to save the model.
        """
        # Make a deep copy of current model instance
        model = copy.deepcopy(self)

        # Reset accumulated values in loss and accuracy objects
        model.loss.new_pass()
        model.accuracy.new_pass()

        # Remove data from the input layer
        # and gradients from the loss object
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        # For each layer remove inputs, output and dinputs properties
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        # Open a file in the binary-write mode and save the model
        with open(path, 'wb') as f:
            pickle.dump(model, f)


    # Loads and returns a model
    @staticmethod
    def load(path: str):
        """
        Loads and returns a model.

        Parameters:
        path (str): The file path to load the model from.

        Returns:
        Model: The loaded model.
        """
        # Open file in the binary-read mode, load a model
        with open(path, 'rb') as f:
            model = pickle.load(f)
        # Return a model
        return model
    
