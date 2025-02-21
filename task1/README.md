
# Task 1: Image Classification with OOP

This project implements image classification on the MNIST dataset using three different models: Random Forest, Feed-Forward Neural Network, and Convolutional Neural Network. Each model is implemented as a separate class, adhering to the `MnistClassifierInterface`, and managed through a unified `MnistClassifier` class.



## Solution Explanation

The core of this solution lies in the object-oriented design. The `MnistClassifierInterface` defines the contract for all models, ensuring they implement `train` and `predict` methods. Each model (`cnn.py`, `neural_network.py`, `random_forest.py`) implements this interface. The `MnistClassifier` class acts as a factory, instantiating the appropriate model based on the provided algorithm name (`cnn`, `rf`, or `nn`). This allows for seamless switching between models without changing the calling code.

*   **Models:**
    *   **Convolutional Neural Network (CNN):** Implemented using PyTorch. Leverages convolutional layers to extract spatial hierarchies from the images.
    *   **Feed-Forward Neural Network (FFNN):** Implemented using PyTorch. A multi-layer perceptron for classifying the images.
    *   **Random Forest:** Implemented using scikit-learn. An ensemble learning method that constructs multiple decision trees during training.

*   **Training:** The `ModelTrainer` class in `training/trainer.py` handles the training and evaluation process. It uses the chosen algorithm from `MnistClassifier` and performs the necessary steps for training, validation, and testing.

*   **Demo:** The `demo.ipynb` notebook demonstrates how to use the `MnistClassifier` to train and predict with different algorithms. It also includes examples of edge cases and how the code handles them.

## Requirements



You can install these dependencies using:

```bash
pip install -r requirements.txt
```

## Running the Code

1.  **Clone the Repository:** Clone this repository to your local machine.
2.  **Install Dependencies:** Install the required libraries using the command above.
3.  **Run the Demo:** Open and run the `demo.ipynb` Jupyter Notebook to see examples of how to use the code.

4.  **Training and Evaluation:** You can train and evaluate the models using the `main.py` script. To run it:

    ```bash
    python main.py
    ```

    This will prompt you to select an algorithm (cnn, rf, or nn). After training, the script will print the evaluation results.



## Further Improvements


*   **Data Augmentation:** Applying data augmentation techniques could further enhance model robustness.
*   **Model Persistence:** Implementing model saving and loading would allow for reuse of trained models without retraining.


