import time
from datetime import datetime
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from data_utils.data_loader import MNISTDataLoader
from models.classifier import MnistClassifier


@dataclass
class TrainingResult:
    """Data class to store training results."""
    accuracy: float
    training_time: float
    evaluation_time: float
    total_time: float
    classification_report: str
    confusion_matrix: np.ndarray


class ModelTrainer:
    """
    Handles model training, evaluation and result visualization.
    """

    VALID_ALGORITHMS = ['cnn', 'rf', 'nn']

    def __init__(self):
        """Initialize the trainer."""
        self.data_loader = MNISTDataLoader()

    def get_algorithm_choice(self) -> str:
        """
        Get user input for algorithm selection.

        Returns:
            str: Selected algorithm name

        Raises:
            ValueError: If invalid algorithm is selected
        """
        print("\nAvailable algorithms:")
        for i, algo in enumerate(self.VALID_ALGORITHMS, 1):
            print(f"{i}. {algo}")

        while True:
            choice = input(
                "\nSelect algorithm number (or type the name): ").strip().lower()

            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(self.VALID_ALGORITHMS):
                    return self.VALID_ALGORITHMS[index]

            if choice in self.VALID_ALGORITHMS:
                return choice

            print(
                f"Error: Please select a valid algorithm (1-3 or {', '.join(self.VALID_ALGORITHMS)})")

    def plot_confusion_matrix(self, confusion_mat: np.ndarray, save_path: str = None) -> None:
        """
        Plot and optionally save confusion matrix visualization.

        Args:
            confusion_mat: Confusion matrix array
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def train_and_evaluate(self, algorithm: str) -> TrainingResult:
        """
        Train and evaluate a model with the specified algorithm.

        Args:
            algorithm: Name of the algorithm to use

        Returns:
            TrainingResult: Object containing training results and metrics
        """
        start_time = time.time()
        print(
            f"\nTraining started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Load data
        print("\nLoading MNIST data...")
        X_train, y_train, X_test, y_test = self.data_loader.load_data()
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

        # Initialize and train model
        print(f"\nInitializing {algorithm} classifier...")
        classifier = MnistClassifier(algorithm=algorithm)

        print("\nTraining model...")
        train_start = time.time()
        classifier.train(X_train, y_train)
        train_time = time.time() - train_start

        # Evaluate model
        print("\nEvaluating model...")
        eval_start = time.time()
        predictions = classifier.predict(X_test)
        eval_time = time.time() - eval_start

        # Calculate metrics
        accuracy = (predictions == y_test).mean()
        total_time = time.time() - start_time
        report = classification_report(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)

        return TrainingResult(
            accuracy=accuracy,
            training_time=train_time,
            evaluation_time=eval_time,
            total_time=total_time,
            classification_report=report,
            confusion_matrix=conf_matrix
        )

    def print_results(self, results: TrainingResult) -> None:
        """
        Print training results in a formatted way.

        Args:
            results: TrainingResult object containing metrics
        """
        print(f"\nAccuracy: {results.accuracy:.4f}")
        print("\nClassification Report:")
        print(results.classification_report)

        print("\nGenerating confusion matrix...")
        self.plot_confusion_matrix(results.confusion_matrix)

        print(f"\nTotal execution time: {results.total_time:.2f} seconds")
        print(f" - Training time: {results.training_time:.2f} seconds")
        print(f" - Evaluation time: {results.evaluation_time:.2f} seconds")
