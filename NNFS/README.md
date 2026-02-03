# Neural Networks from Scratch (NNFS)

This project implements a neural network from scratch using Python and NumPy, following the principles of deep learning. It includes a complete pipeline for loading data, preprocessing, defining a model architecture, training, and making predictions on the Fashion MNIST dataset.

## Features

-   **Modular Architecture**: Clean separation of concerns with dedicated modules for:
    -   **Layers**: Dense (Fully Connected), Dropout.
    -   **Activations**: ReLU, Softmax, Sigmoid, Linear.
    -   **Loss Functions**: Categorical Crossentropy, Mean Squared Error, Binary Crossentropy.
    -   **Optimizers**: Adam, SGD, Adagrad, RMSprop.
-   **Dataset Handling**: Automated downloading and loading of the **Fashion MNIST** dataset.
-   **Training Pipeline**: Complete forward and backward pass implementation with backpropagation.
-   **Model Management**: Save and load model parameters and architecture.
-   **Visualization**: Real-time progress updates and prediction visualization using Matplotlib and OpenCV.

## Project Structure

The project is organized into the following directories:

| Directory/File | Description |
| :--- | :--- |
| `accuracy/` | Classes for calculating model accuracy (Regression, Categorical). |
| `activation/` | Activation functions (ReLU, Softmax, Sigmoid, etc.). |
| `data/` | Utilities for loading and handling datasets. |
| `layers/` | Implementation of neural network layers (Dense, Dropout). |
| `loss/` | Loss functions (Categorical Crossentropy, etc.). |
| `model/` | Core `Model` class handling the training and prediction loops. |
| `optimizer/` | Optimization algorithms (Adam, SGD, etc.). |
| `Data/` | Stores the downloaded Fashion MNIST dataset (created at runtime). |
| `Models/` | Stores saved model files (`.pickle`) (created after training). |
| `Plots/` | Stores generated visualization plots (created after prediction). |
| `main.py` | The entry point script for training and testing the model. |

## Project Flow

The `main.py` script orchestrates the entire process:

1.  **Data Loading**: Downloads (if necessary) and loads the Fashion MNIST dataset.
2.  **Preprocessing**: Shuffles the dataset, reshapes images, and scales pixel values to [-1, 1].
3.  **Model Creation**: 
    -   Initializes a defined architecture (e.g., Dense -> ReLU -> Dense -> ReLU -> Dense -> Softmax).
    -   Configures Loss, Optimizer (Adam), and Accuracy metrics.
4.  **Training**: Runs the training loop for a specified number of epochs, performing forward and backward passes and updating weights.
5.  **Saving**: Saves the trained model parameters to the `Models/` directory.
6.  **Prediction**: 
    -   Loads a test image.
    -   Loads the trained model.
    -   Runs inference to predict the clothing item.
    -   Visualizes the result and confidence levels.

## Prerequisites & Installation

This project uses `uv` for dependency management.

### 1. Install `uv`
If you haven't installed `uv` yet, please follow the [official installation guide](https://github.com/astral-sh/uv).

### 2. Install Dependencies
Navigate to the project directory and sync the dependencies. This will create a virtual environment and install all required packages defined in `pyproject.toml`.

```bash
uv sync
```

## Usage

### Training and Prediction
To run the project, execute the `main.py` script using `uv run`. This ensures the script runs within the managed environment with all dependencies available.

```bash
uv run main.py
```

**What to expect:**
-   The script will download the Fashion MNIST data (first run only).
-   It will train the model for 5 epochs, showing loss and accuracy progress.
-   After training, it will save the model.
-   It will then perform a prediction on a sample test image (`Data/Testing/tshirt_test1.png`) and display the result along with a confidence bar chart.
