from sklearn.ensemble import RandomForestClassifier as SklearnRF
import numpy as np

from .interface import BaseClassifier


class RandomForestClassifier(BaseClassifier):
    """
    Random Forest implementation for MNIST classification using scikit-learn.
    """

    def __init__(self, n_estimators: int = 150):
        """
        Initialize the random forest classifier.

        Args:
            n_estimators: Number of trees in the forest
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.model = SklearnRF(n_estimators=self.n_estimators)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> 'RandomForestClassifier':
        """
        Train the random forest model.

        Args:
            X_train: Training images (batch_size, channels, height, width)
            y_train: Training labels

        Returns:
            self: Trained classifier instance
        """
        self._validate_input_shape(X_train, (1, 28, 28))

        # Flatten images for sklearn
        X_train_flat = X_train.reshape(X_train.shape[0], -1)

        self.model.fit(X_train_flat, y_train)
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

        # Flatten images for sklearn
        X_flat = X.reshape(X.shape[0], -1)
        return self.model.predict(X_flat)
