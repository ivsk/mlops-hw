import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os
import argparse
import boto3
from pathlib import Path

# SageMaker specific paths
INPUT_PATH = "/opt/ml/input/data"
OUTPUT_PATH = "/opt/ml/output"
MODEL_PATH = "/opt/ml/model"
PARAM_PATH = "/opt/ml/input/config/hyperparameters.json"


def parse_args():
    parser = argparse.ArgumentParser()

    # SageMaker specific arguments
    parser.add_argument("--model-dir", type=str, default=MODEL_PATH)
    parser.add_argument(
        "--train", type=str, default=os.path.join(INPUT_PATH, "training")
    )

    # Model hyperparameters
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)

    return parser.parse_args()


def load_data(file_path, is_train=True):
    """Load data from text file"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        header = lines[0].strip().split(" ::: ")
        print(f"Header for {file_path}: {header}")
        for line in lines[1:]:  # Skip header
            parts = line.strip().split(" ::: ")
            if is_train:
                if len(parts) == 4:
                    data.append(
                        {
                            "ID": parts[0],
                            "TITLE": parts[1],
                            "GENRE": parts[2],
                            "DESCRIPTION": parts[3],
                        }
                    )
                else:
                    print(f"Skipping malformed line in train data: {line.strip()}")
            else:
                if len(parts) == 3:
                    data.append(
                        {"ID": parts[0], "TITLE": parts[1], "DESCRIPTION": parts[2]}
                    )
                else:
                    print(f"Skipping malformed line in test data: {line.strip()}")
    return pd.DataFrame(data)


class GenreDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def compute_metrics(pred):
    """Compute metrics for evaluation"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def main():
    args = parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load training data
    train_file = os.path.join(args.train, "train_data.txt")
    try:
        train_df_full = load_data(train_file, is_train=True)
        print(f"Loaded {len(train_df_full)} training samples.")
    except FileNotFoundError:
        print(f"Error: {train_file} not found.")
        return

    if train_df_full.empty:
        print("Training data is empty. Exiting.")
        return

    # Preprocessing
    train_df_full["TEXT"] = (
        train_df_full["TITLE"] + " [SEP] " + train_df_full["DESCRIPTION"]
    )

    # Create label mappings
    unique_genres = sorted(list(train_df_full["GENRE"].unique()))
    genre_to_id = {genre: i for i, genre in enumerate(unique_genres)}
    id_to_genre = {i: genre for genre, i in genre_to_id.items()}
    num_labels = len(unique_genres)

    print(f"Genre to ID mapping: {genre_to_id}")
    print(f"Number of unique genres: {num_labels}")

    train_df_full["labels"] = train_df_full["GENRE"].map(genre_to_id)

    # Split data
    if len(train_df_full) > 1:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_df_full["TEXT"].tolist(),
            train_df_full["labels"].tolist(),
            test_size=0.15,
            random_state=42,
            stratify=train_df_full["labels"].tolist(),
        )
    else:
        train_texts = train_df_full["TEXT"].tolist()
        train_labels = train_df_full["labels"].tolist()
        val_texts = train_texts
        val_labels = train_labels

    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")

    # Tokenization
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)

    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, max_length=args.max_length
    )
    val_encodings = tokenizer(
        val_texts, truncation=True, padding=True, max_length=args.max_length
    )

    # Create datasets
    train_dataset = GenreDataset(train_encodings, train_labels)
    val_dataset = GenreDataset(val_encodings, val_labels)

    # Load model
    model = BertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id_to_genre,
        label2id=genre_to_id,
    )
    model.to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.model_dir, "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        save_total_limit=1,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\n--- Starting Training ---")
    trainer.train()
    print("\n--- Training Finished ---")

    # Evaluate
    print("\n--- Evaluating Best Model ---")
    eval_results = trainer.evaluate(eval_dataset=val_dataset)
    print("Validation Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")

    # Save model and tokenizer
    print(f"\nSaving model to {args.model_dir}")
    trainer.save_model(args.model_dir)

    # Save label mappings
    label_map_path = os.path.join(args.model_dir, "label_mappings.pth")
    torch.save(
        {
            "genre_to_id": genre_to_id,
            "id_to_genre": id_to_genre,
            "num_labels": num_labels,
        },
        label_map_path,
    )
    print(f"Label mappings saved to {label_map_path}")

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
