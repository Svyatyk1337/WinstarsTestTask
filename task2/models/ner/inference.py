import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import sys


def main():
    if len(sys.argv) != 2:
        print("Usage: python inference.py \"Input text\"")
        sys.exit(1)

    text = sys.argv[1]

    model_dir = "animal_ner_model"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize the NER pipeline
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

    def extract_animals(text):
        """Extract animal names from the input text."""
        ner_results = ner_pipeline(text)
        animals = [entity["word"]
                   for entity in ner_results if entity["entity"] == "LABEL_1"]
        return animals

    # Extract animals from the input text
    animals = extract_animals(text)

    print("Found animals:", animals)


if __name__ == "__main__":
    main()
