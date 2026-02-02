import os
import cv2
import numpy as np
from tqdm import tqdm 
import urllib.request
from zipfile import ZipFile

from banners import datalog

def show_banner(text, msg):
    lines = text.strip().split("\n")

    width = max(len(line) for line in lines)
    border = "─" * (width + 4)

    print(f"┌{border}┐")

    for line in lines:
        print(f"│  {line.ljust(width)}  │")

    print(f"└{border}┘")

    print(f"\n> {msg}\n")

# Data Class for creating and loading data
class Data:
    def __init__(self):
        self.url = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
    # Loads a MNIST dataset
    def load_mnist_dataset(self, dataset: str, path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads a MNIST dataset from the specified directory.

        Parameters:
        dataset (str): The dataset type ('train' or 'test').
        path (str): The path to the dataset directory.

        Returns:
        tuple: A tuple containing the images (X) and labels (y) as numpy arrays.
        """
        # Scan all the directories and create a list of labels
        labels = os.listdir(os.path.join(path, dataset))

        # Create lists for samples and labels
        X, y = [], []

        # For each label folder
        for label in labels:
            # And for each image in given folder
            for file in tqdm(os.listdir(os.path.join(path, dataset, label)), desc=f"Label {label}:"):
                # Read the image
                image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
                # And append it and a label to the lists
                X.append(image)
                y.append(label)

        # Convert the data to proper numpy arrays and return
        return np.array(X), np.array(y).astype('uint8')


    # MNIST dataset (train + test)
    def create_fashion_mnist(self, file_path: str, folder_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates the Fashion MNIST dataset by downloading and extracting it if necessary.

        Parameters:
        file_path (str): The path to the zip file.
        folder_path (str): The folder where the dataset will be extracted.

        Returns:
        tuple: A tuple containing the training images, training labels, test images, and test labels as numpy arrays.
        """
        print("\n====\tStarting Download\t====\n")
        # Download the dataset
        if not os.path.isfile(file_path):
            print(f"Downloading `{self.url}` and saving as `{file_path}`\n")
            urllib.request.urlretrieve(self.url, file_path)

        # Extract the files into folder
        with ZipFile(file_path) as zip_images:
            members = zip_images.infolist()
            print(f"Unzipping `{file_path}` into `{folder_path}`\n")
            for member in tqdm(members, desc="Extracting"):
                zip_images.extract(member, folder_path)
        print("\n====\tDone!!\t====\n")

        show_banner(datalog, "Loading Fashion MNIST Dataset")

        # Load both sets separately
        print("\n====\tLoad Training Data\t====\n")
        X, y = self.load_mnist_dataset('train', folder_path)
        print("\n====\tLoad Test Data\t====\n")
        X_test, y_test = self.load_mnist_dataset('test', folder_path)

        # And return all the data
        return X, y, X_test, y_test

    # Create demo spiral data
    def create_spiral_data(self, samples: int, classes: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Creates demo spiral data.

        Parameters:
        samples (int): The number of samples to generate.
        classes (int): The number of classes.

        Returns:
        tuple: A tuple containing the generated data (X) and labels (y) as numpy arrays.
        """
        from nnfs.datasets import spiral_data

        # Create spiral data
        X, y = spiral_data(samples, classes)
        return X, y

    # Create demo vertical data
    def create_vertical_data(self, samples: int, classes: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Creates demo vertical data.

        Parameters:
        samples (int): The number of samples to generate.
        classes (int): The number of classes.

        Returns:
        tuple: A tuple containing the generated data (X) and labels (y) as numpy arrays.
        """
        from nnfs.datasets import vertical_data

        # Create vertical data
        X, y = vertical_data(samples, classes)
        return X, y

    # Create XOR demo
    def create_xor_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Creates XOR demo data.

        Returns:
        tuple: A tuple containing the generated data (X) and labels (y) as numpy arrays.
        """
        # Create xor data
        X = np.array([[0,0], [0,1], [1,0], [1,1]])
        y = np.array([0, 1, 1, 0])
        return X, y

