import os
import cv2
import nnfs
import numpy as np
from colorama import Fore, Style, init
from matplotlib import pyplot as plt

from data.data import Data
from model.model import Model
from layers.dense import Dense
from activation.relu import ReLU
from activation.softmax import Softmax
from loss.classification import CategoricalCrossentropy
from optimizer.adam import Adam
from accuracy.classification import Categorical

from banners import (banner, model_banner, trainer, predict_banner)

# Define constants for file and folder names
DIR_PATH = "/home/abhijit/Use Directory/Python/DL from scratch/NNFS"
ZIP_FILE = 'Data/fashion_mnist_images.zip'
FILE = os.path.join(DIR_PATH, ZIP_FILE)
FOLDER = os.path.join(DIR_PATH, 'Data/fashion_mnist_images')
MODEL_PATH = os.path.join(DIR_PATH, 'Models/fashion_mnist_model.pickle')
PARAM_PATH = os.path.join(DIR_PATH, 'Models/fashion_mnist_model_params.pickle')
PLOT_PATH = os.path.join(DIR_PATH, 'Plots')
TEST_IMAGE1 = os.path.join(DIR_PATH, 'Data/Testing/tshirt_test1.png')
TEST_IMAGE2 = os.path.join(DIR_PATH, 'Data/Testing/pants_test2.png')


def main():
    nnfs.init()

    """ <<=============================== Data Loading ==============================>> """
    # Load the data
    data = Data()
    os.makedirs(FOLDER, exist_ok=True)
    X, y, X_test, y_test = data.create_fashion_mnist(FILE, FOLDER)

    """ <<============================ Data Preprocessing =============================>> """
    # Shuffle the data
    keys = np.array(range(X.shape[0]))
    np.random.shuffle(keys)
    X = X[keys]
    y = y[keys]

    # Scales and reshapes samples
    X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5
    
    print(f"\n{Fore.GREEN}Data preprocessing completed successfully.{Style.RESET_ALL}\n")

    """ <<=============================== Model Creation ==============================>> """
    show_banner(model_banner, "Created Model")
    ## Create the model
    model = Model()

    # Add layers
    model.add(Dense(X.shape[1], 64))
    model.add(ReLU())
    model.add(Dense(64, 64))
    model.add(ReLU())
    model.add(Dense(64, 10))
    model.add(Softmax())

    # Set the loss and optimizer
    model.set(
        loss = CategoricalCrossentropy(),
        optimizer = Adam(decay=5e-5),
        accuracy=Categorical()
    )

    # Finalize model
    model.finalize()
    # Model Summary
    model.summary()

    """ <<=============================== Model Training ==============================>> """
    show_banner(trainer, "Training Model")
    # Train the model
    model.train(X, y, validation_data=(X_test, y_test), epochs=5, batch_size=128, print_every=100)
    # Save the model
    model.save_parameters(PARAM_PATH)
    model.save(MODEL_PATH)
    print(f"\n{Fore.GREEN}Model saved to:{Style.RESET_ALL} {MODEL_PATH}\n")


    """ <<=============================== Model Prediction ==============================>> """
    show_banner(predict_banner, "Model Prediction")
    # Label index to label name relation
    fashion_mnist_labels = {
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
    }

    # Read an image
    image_data = cv2.imread(TEST_IMAGE1, cv2.IMREAD_GRAYSCALE)
    if image_data is None:
        raise Exception("Image not found!")
    else:
        print(f"{Fore.GREEN}Image loaded successfully from:{Style.RESET_ALL} {TEST_IMAGE1}")

    img_data = image_data.copy()

    # Resize to the same size as Fashion MNIST images
    image_data = cv2.resize(image_data, (28, 28))

    # Invert image colors
    image_data = 255 - image_data

    # Reshape and scale pixel data
    image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

    # Load the model
    model = Model.load(MODEL_PATH)
    
    # Predict on the image
    confidences = model.predict(image_data)

    # Get prediction instead of confidence levels
    predictions = model.output_layer_activation.predictions(confidences)

    # Get label name from label index
    prediction = fashion_mnist_labels[predictions[0]]

    # Display the image
    print(f"\n{Style.BRIGHT}{Fore.MAGENTA}Predicted Label=> {prediction}{Style.RESET_ALL}\n")
    # Plot the image with prediction
    plt.figure(figsize=(5,5))
    plt.imshow(img_data)
    plt.axis('off')
    plt.title(f'Predicted: {prediction}')
    plt.savefig(os.path.join(PLOT_PATH, 'prediction.png'))
    plt.show()

    # Plot confidence levels
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.title('Input Image')
    plt.axis('off')
    plt.imshow(img_data, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('Confidence Levels')
    plt.bar(list(fashion_mnist_labels.values()), confidences[0])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, 'confidence_levels.png'))
    plt.show()

    """ <<=============================== End of Script ==============================>> """


def show_banner(text, msg):
    lines = text.strip().split("\n")

    width = max(len(line) for line in lines)
    border = "─" * (width + 4)

    print(f"┌{border}┐")

    for line in lines:
        print(f"│  {line.ljust(width)}  │")

    print(f"└{border}┘")

    print(f"\n> {msg}\n")


if __name__ == "__main__":
    init(autoreset=True)
    show_banner(banner, "Welcome to NNFS Trainer")
    main()

    