from abc import ABC, abstractmethod
import numpy as np

class BaseClassifier(ABC):
    """
    Abstract base class for all MNIST classifiers.
    Defines the common interface and shared functionality.
    """
    
    def __init__(self):
        self.is_trained = False
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> 'BaseClassifier':
        """
        Train the classifier on the given data.
        
        Args:
            X_train: Training images array
            y_train: Training labels array
            
        Returns:
            self: The trained classifier instance
            
        Raises:
            InvalidInputShapeError: If input data has incorrect shape
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes for given images.
        
        Args:
            X: Input images to classify
            
        Returns:
            numpy.ndarray: Predicted class labels
            
        Raises:
            ModelNotTrainedError: If model hasn't been trained
            InvalidInputShapeError: If input data has incorrect shape
        """
        pass
    
    def _validate_input_shape(self, X: np.ndarray, expected_shape: tuple) -> None:
        """
        Validate input data shape.
        
        Args:
            X: Input data to validate
            expected_shape: Expected shape tuple (excluding batch dimension)
            
        Raises:
            InvalidInputShapeError: If shape is invalid
        """
        if len(X.shape) != len(expected_shape) + 1 or X.shape[1:] != expected_shape:
            raise Exception(
                f"Expected input shape: (batch_size, {expected_shape}), got: {X.shape}"
            )