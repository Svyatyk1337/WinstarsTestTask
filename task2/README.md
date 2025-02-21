# Animal Detection & NER Pipeline

This project implements a pipeline that combines Named Entity Recognition (NER) for animal detection in text and Image Classification for animal detection in images. The system verifies if the animal mentioned in the text matches the animal present in the provided image.

## Solution Explanation

First, I built a computer vision model to classify animals in images. Then, I developed an NER (Named Entity Recognition) model to detect animal mentions in text. Finally, I combined both models into a unified system that can analyze both images and text, verifying whether the detected animal in the text matches the one in the image.

*   **Models:**
    *   **Image Classification Model:** Implemented using ResNet50. Detects animals in images and classifies them into predefined categories.
    *   **NER Model:** Implemented using a fine-tuned BERT model. Extracts animal mentions from text and classifies them using BIO tagging.

*  **Training**: Each model has its own train.py script for training. The image classification model is trained using image_classifier/train.py, while the NER model is trained using ner_model/train.py. These scripts handle data loading, training, validation, and model saving.

* **Inference:** For making predictions, each model has an inference.py script. The image classifier uses image_classifier/inference.py, and the NER model uses ner_model/inference.py. These scripts load the trained models and process new input data to generate predictions.

*   **Demo:** The `demo.ipynb` notebook demonstrates how to use the AnimalDetector to process text and images.

## Dataset Analysis

### Image Classification Dataset

The image classification dataset contains images of 10 animal classes:
- butterfly
- cat
- cock
- cow
- dog
- elephant
- horse
- sheep
- spider
- squirrel

### NER Dataset

The NER dataset analysis is documented in `ner_dataset_eda.ipynb`. The dataset was processed to identify animal mentions in text.

## Models

### Image Classification Model

Implementation details:
- Architecture: ResNet50
- Input size: 224x224 pixels
- Output: 10 classes (softmax)
- Training parameters:
  - Optimizer: Adam
  - Learning rate: 0.001
  - Batch size: 128
  - Epochs: 40
- Result:
  - Accuracy: 87 %
  - Macro precision: 0.8740
  - Macro recall: 0.8506
  - Macro F1-score: 0.8602

Data augmentation:
- Random horizontal flip
- Random rotation
- Random crop

### NER Model

Implementation details:
- Base model: BERT (bert-base-uncased)
- Fine-tuned for token classification
- Training parameters:
  - Learning rate: 2e-5
  - Batch size: 16
  - Epochs: 10
- Label scheme: BIO tagging

## Setup Instructions

1. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Running the Code

1. **Clone the Repository:** Clone this repository to your local machine.
2. **Install Dependencies:** Install the required libraries using the command above.
3. **Run the Demo:** Open and run the `demo.ipynb` Jupyter Notebook to see examples of how to use the code.
4. **Run the Complete Pipeline:**
```bash
python main.py 'Text' /path/to/image.jpg
```
## Example
![image](https://github.com/user-attachments/assets/4818b648-e337-460f-8f2c-3c6423a632c0)





