import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F

from .interface import BaseClassifier


import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    Convolutional Neural Network architecture for MNIST classification.
    Implements the architecture from the Keras model with BatchNormalization.
    """

    def __init__(self):
        """Initialize CNN layers."""
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


class CNNClassifier(BaseClassifier):
    """
    CNN implementation for MNIST classification.
    """

    def __init__(self, epochs: int = 35, batch_size: int = 64, learning_rate: float = 0.001):
        """
        Initialize the CNN classifier.

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

    def _build_model(self) -> None:
        """Initialize the CNN model, loss function, and optimizer."""
        self.model = CNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> 'CNNClassifier':
        """
        Train the CNN model.

        Args:
            X_train: Training images (batch_size, channels, height, width)
            y_train: Training labels

        Returns:
            self: Trained classifier instance
        """
        self._validate_input_shape(X_train, (1, 28, 28))

        # Prepare data
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model if needed
        if self.model is None:
            self._build_model()

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in dataloader:
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
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False)

        # Make predictions
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model(batch[0])
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())

        return np.array(predictions)
