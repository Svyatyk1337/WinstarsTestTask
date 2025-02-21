import numpy as np
from torchvision import datasets, transforms
from typing import Tuple


class MNISTDataLoader:
    """
    Handles loading and preprocessing of MNIST dataset.
    """

    def __init__(self, data_root: str = "./data"):
        """
        Initialize the data loader.

        Args:
            data_root: Directory to store MNIST data
        """
        self.data_root = data_root

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess MNIST dataset.

        Returns:
            Tuple containing:
            - X_train: Training images
            - y_train: Training labels
            - X_test: Test images
            - y_test: Test labels
        """
        # Load datasets
        mnist_train = datasets.MNIST(
            root=self.data_root,
            train=True,
            download=True,

        )
        mnist_test = datasets.MNIST(
            root=self.data_root,
            train=False,
            download=True,

        )

        # Convert to numpy arrays
        X_train = mnist_train.data.numpy().astype(np.float32) / 255.0
        y_train = mnist_train.targets.numpy()
        X_test = mnist_test.data.numpy().astype(np.float32) / 255.0
        y_test = mnist_test.targets.numpy()

        # Add channel dimension
        X_train = X_train[:, np.newaxis, :, :]
        X_test = X_test[:, np.newaxis, :, :]

        return X_train, y_train, X_test, y_test
