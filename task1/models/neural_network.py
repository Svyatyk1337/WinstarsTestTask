import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from .interface import BaseClassifier


class NeuralNetwork(nn.Module):
    """
    PyTorch neural network architecture for MNIST classification.
    Uses a deep fully-connected network with ReLU activations.
    """

    def __init__(self, input_size: int, hidden_sizes: list = [2000, 1500, 1000, 500], num_classes: int = 10):
        """
        Initialize neural network layers.

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes
        """
        super().__init__()

        layers = []
        prev_size = input_size

        # Build hidden layers
        for size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU()
            ])
            prev_size = size

        # Add output layer
        layers.append(nn.Linear(prev_size, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.model(x)


class NeuralNetworkClassifier(BaseClassifier):
    """
    Neural Network implementation for MNIST classification using PyTorch.
    """

    def __init__(self, epochs: int = 40, batch_size: int = 64, learning_rate: float = 0.001):
        """
        Initialize the neural network classifier.

        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
        """
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def _build_model(self, input_size: int) -> None:
        """
        Initialize the PyTorch model, loss function, and optimizer.

        Args:
            input_size: Number of input features
        """
        self.model = NeuralNetwork(input_size).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> 'NeuralNetworkClassifier':
        """
        Train the neural network.

        Args:
            X_train: Training images (batch_size, channels, height, width)
            y_train: Training labels

        Returns:
            self: Trained classifier instance
        """
        self._validate_input_shape(X_train, (1, 28, 28))

        # Prepare data
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_train_flat = X_train_tensor.view(X_train_tensor.size(0), -1)

        # Initialize model if needed
        if self.model is None:
            self._build_model(X_train_flat.shape[1])

        # Create data loader
        dataset = TensorDataset(X_train_flat, y_train_tensor)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        self.is_trained = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes for input images.

        Args:
            X: Input images to classify (batch_size, channels, height, width)

        Returns:
            numpy.ndarray: Predicted class labels

        Raises:
            ModelNotTrainedError: If model hasn't been trained
        """
        if not self.is_trained:
            raise Exception("Model must be trained before making predictions")

        self._validate_input_shape(X, (1, 28, 28))

        # Prepare data
        X_tensor = torch.FloatTensor(X).to(self.device)
        X_flat = X_tensor.view(X_tensor.size(0), -1)

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_flat)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().numpy()
