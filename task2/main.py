import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from torchvision import transforms
from PIL import Image
import sys
from models.classifier.resnet import ResNet50

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the NER model
NER_MODEL_DIR = "./models/ner/animal_ner_model"
try:
    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_DIR)
    ner_model = AutoModelForTokenClassification.from_pretrained(
        NER_MODEL_DIR).to(device)
    ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer)
except Exception as e:
    print(f"Error loading NER model: {e}")
    sys.exit(1)

# Load the image classification model
CLASS_MODEL_PATH = "./models/classifier/classification_model/model.pth"
try:
    classification_model = ResNet50(3, 10).to(device)
    classification_model.load_state_dict(
        torch.load(CLASS_MODEL_PATH, map_location=device))
    classification_model.eval()
except Exception as e:
    print(f"Error loading classification model: {e}")
    sys.exit(1)

# Mapping of class IDs to animal names
class_labels = {
    0: "butterfly",
    1: "cat",
    2: "cock",
    3: "cow",
    4: "dog",
    5: "elephant",
    6: "horse",
    7: "sheep",
    8: "spider",
    9: "squirrel"
}

# Image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def extract_animals_from_text(text):
    """Extracts animal names from the input text using NER."""
    ner_results = ner_pipeline(text)

    animals = set()
    for entity in ner_results:
        if entity["entity"] == "LABEL_1":
            animals.add(entity["word"].lower())

    return animals


def classify_animal_image(image_path):
    """Classifies the animal in the input image."""
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = classification_model(image)
        _, predicted_class = torch.max(outputs, 1)

    return class_labels.get(predicted_class.item())


def verify_text_and_image(text, image_path):
    """Verifies if the animal in the text matches the animal in the image (at least one match)."""
    animals_in_text = extract_animals_from_text(text)
    predicted_animal = classify_animal_image(image_path)

    

    if predicted_animal is None:
        return False

    match_found = any(predicted_animal.lower() == animal.lower()
                      for animal in animals_in_text)

    return True if match_found else False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py 'Text' /path/to/image.jpg")
        sys.exit(1)

    input_text = sys.argv[1]
    image_path = sys.argv[2]

    result = verify_text_and_image(input_text, image_path)
    print(f"Result: {result}")
