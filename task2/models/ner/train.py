import os
import json
import torch
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import Dataset, DatasetDict

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

def load_dataset(dataset_path):
    """Load dataset from a JSON file."""
    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return DatasetDict({
        'train': Dataset.from_dict({
            'id': [ex['id'] for ex in data['train']],
            'tokens': [ex['tokens'] for ex in data['train']],
            'entities': [ex['entities'] for ex in data['train']]
        }),
        'validation': Dataset.from_dict({
            'id': [ex['id'] for ex in data['validation']],
            'tokens': [ex['tokens'] for ex in data['validation']],
            'entities': [ex['entities'] for ex in data['validation']]
        }),
        'test': Dataset.from_dict({
            'id': [ex['id'] for ex in data['test']],
            'tokens': [ex['tokens'] for ex in data['test']],
            'entities': [ex['entities'] for ex in data['test']]
        })
    })

def tokenize_and_align_labels(examples, tokenizer, tag_mapping, max_length):
    """Tokenize inputs and align labels for token classification."""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    labels = []
    for i, entities in enumerate(examples["entities"]):
        tags = ["O"] * len(tokenized_inputs["input_ids"][i])
        word_ids = tokenized_inputs.word_ids(batch_index=i)

        for entity in entities:
            entity_start, entity_end, entity_type = entity["start"], entity["end"], entity["label"]
            for word_idx in range(entity_start, entity_end):
                token_idxs = [idx for idx, word_id in enumerate(
                    word_ids) if word_id == word_idx]
                if token_idxs:
                    tags[token_idxs[0]] = f"B-{entity_type}"
                    for token_idx in token_idxs[1:]:
                        tags[token_idx] = f"I-{entity_type}"

        label_ids = [tag_mapping.get(tag, 0) for tag in tags]
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def main():
    parser = argparse.ArgumentParser(description='Train a Named Entity Recognition (NER) model.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset JSON file')
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased', help='Pretrained model name')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the trained model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio for learning rate scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tag_mapping = {"O": 0, "B-ANIMAL": 1, "I-ANIMAL": 2}
    
    datasets = load_dataset(args.dataset_path)
    tokenized_datasets = datasets.map(lambda x: tokenize_and_align_labels(x, tokenizer, tag_mapping, args.max_length), batched=True)
    
    # Load model
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(tag_mapping))
    model.to(device)

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )

    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model successfully saved to {args.output_dir}")

if __name__ == "__main__":
    main()
