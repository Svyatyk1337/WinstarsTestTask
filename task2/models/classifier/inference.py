import torch
from torchvision import transforms
from PIL import Image
import sys
from resnet import ResNet50

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
MODEL_PATH = "./classification_model/model.pth"
model = ResNet50(3, 10).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Class labels dictionary
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


def predict(image_path):
    """Predicts the class of the input image."""
    try:
        image = Image.open(image_path).convert(
            "RGB")  # Open and convert to RGB
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    # Add batch dimension and move to device
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        # Get the class with the highest probability
        _, predicted_class = torch.max(outputs, 1)

    return predicted_class.item()  # Return the predicted class ID


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference.py /path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    class_id = predict(image_path)

    if class_id is not None:
        predicted_class_name = class_labels.get(class_id)
        if predicted_class_name:
            print(f"Predicted class: {predicted_class_name}")
        else:
            print(f"Predicted class ID: {class_id} (Label name not found)")
