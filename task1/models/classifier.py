from .interface import BaseClassifier
from .random_forest import RandomForestClassifier
from .neural_network import NeuralNetworkClassifier
from .cnn import CNNClassifier


class MnistClassifier:
    def __init__(self, algorithm='cnn'):
        """
        Initialize the MNIST classifier with the specified algorithm.

        Parameters:
        algorithm (str): The algorithm to use - 'cnn', 'rf', or 'nn'

        Raises:
        ValueError: If the algorithm is not one of the supported types
        """
        if algorithm == 'rf':
            self.model = RandomForestClassifier()
        elif algorithm == 'nn':
            self.model = NeuralNetworkClassifier()
        elif algorithm == 'cnn':
            self.model = CNNClassifier()
        else:
            raise ValueError(
                f"Unknown algorithm: {algorithm}. Expected one of: 'cnn', 'rf', 'nn'")

    def train(self, X_train, y_train):
        """
        Train the classifier on the given data.

        Parameters:
        X_train (numpy.ndarray): Training images
        y_train (numpy.ndarray): Training labels

        Returns:
        self: The trained classifier
        """
        self.model.train(X_train, y_train)
        return self

    def predict(self, X):
        """
        Predict the class labels for the input samples.

        Parameters:
        X (numpy.ndarray): Test images

        Returns:
        numpy.ndarray: Predicted class labels
        """
        return self.model.predict(X)
