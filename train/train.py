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
import json
from pathlib import Path
from datetime import datetime

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

    # Model Registry arguments
    parser.add_argument(
        "--model-package-group-name", type=str, default="bert-genre-classifier"
    )
    parser.add_argument(
        "--model-approval-status",
        type=str,
        default="PendingManualApproval",
        choices=["Approved", "Rejected", "PendingManualApproval"],
    )

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


def create_model_package_group(sagemaker_client, group_name, description):
    """Create model package group if it doesn't exist"""
    try:
        sagemaker_client.describe_model_package_group(ModelPackageGroupName=group_name)
        print(f"✅ Model package group '{group_name}' already exists")
        return True
    except sagemaker_client.exceptions.ClientError as e:
        if "does not exist" in str(e).lower():
            try:
                # Add tags to the model package group instead of individual packages
                sagemaker_client.create_model_package_group(
                    ModelPackageGroupName=group_name,
                    ModelPackageGroupDescription=description,
                    Tags=[
                        {"Key": "Project", "Value": "bert-genre-classifier"},
                        {"Key": "Framework", "Value": "PyTorch"},
                        {"Key": "ModelType", "Value": "TextClassification"},
                        {"Key": "CreatedBy", "Value": "SageMakerTraining"},
                        {"Key": "Timestamp", "Value": datetime.utcnow().isoformat()},
                    ],
                )
                print(f"✅ Created model package group '{group_name}' with tags")
                return True
            except Exception as create_error:
                print(f"❌ Error creating model package group: {create_error}")
                return False
        else:
            print(f"❌ Error checking model package group: {e}")
            return False


def register_model_version(
    args, model_artifacts_url, evaluation_results, training_metadata
):
    """Register model version in SageMaker Model Registry"""

    print("\n🔄 Registering model in SageMaker Model Registry...")

    # Get training job name from environment (set by SageMaker)
    training_job_name = os.environ.get("SM_TRAINING_JOB_NAME", "unknown-job")

    try:
        sagemaker_client = boto3.client("sagemaker")

        # Create model package group if it doesn't exist
        group_created = create_model_package_group(
            sagemaker_client,
            args.model_package_group_name,
            "BERT Genre Classifier model versions",
        )

        if not group_created:
            print(
                "❌ Failed to create/verify model package group. Skipping model registration."
            )
            return None

        # Prepare model metrics for registry
        model_metrics = {
            "ModelQuality": {
                "Statistics": {
                    "ContentType": "application/json",
                    "S3Uri": f"{model_artifacts_url.replace('/model.tar.gz', '')}/evaluation_results.json",
                }
            }
        }

        # Save evaluation results to S3 for model metrics
        save_evaluation_to_s3(
            evaluation_results, training_metadata, model_artifacts_url
        )

        # Prepare inference specification
        inference_specification = {
            "Containers": [
                {
                    "Image": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04-sagemaker",
                    "ModelDataUrl": model_artifacts_url,
                    "Environment": {
                        "SAGEMAKER_PROGRAM": "inference.py",
                        "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/code",
                        "MODEL_NAME": args.model_name,
                        "MAX_LENGTH": str(args.max_length),
                    },
                }
            ],
            "SupportedContentTypes": ["application/json", "text/csv"],
            "SupportedResponseMIMETypes": ["application/json"],
            "SupportedRealtimeInferenceInstanceTypes": [
                "ml.t2.medium",
                "ml.m5.large",
                "ml.m5.xlarge",
                "ml.c5.large",
                "ml.c5.xlarge",
            ],
            "SupportedTransformInstanceTypes": ["ml.m5.large", "ml.m5.xlarge"],
        }

        # Create model package (remove Tags from here)
        model_package_input = {
            "ModelPackageGroupName": args.model_package_group_name,
            "ModelPackageDescription": f"BERT Genre Classifier trained on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
            "ModelApprovalStatus": args.model_approval_status,
            "InferenceSpecification": inference_specification,
            "ModelMetrics": model_metrics,
            "CustomerMetadataProperties": {
                "TrainingJobName": training_job_name,
                "ModelName": args.model_name,
                "Epochs": str(args.num_train_epochs),
                "BatchSize": str(args.per_device_train_batch_size),
                "LearningRate": str(args.learning_rate),
                "WeightDecay": str(args.weight_decay),
                "WarmupSteps": str(args.warmup_steps),
                "MaxLength": str(args.max_length),
                "Accuracy": f"{evaluation_results.get('eval_accuracy', 0):.4f}",
                "F1Score": f"{evaluation_results.get('eval_f1', 0):.4f}",
                "Precision": f"{evaluation_results.get('eval_precision', 0):.4f}",
                "Recall": f"{evaluation_results.get('eval_recall', 0):.4f}",
                "TrainingLoss": f"{evaluation_results.get('train_loss', 0):.4f}",
                "ValidationLoss": f"{evaluation_results.get('eval_loss', 0):.4f}",
                "Framework": "PyTorch",
                "FrameworkVersion": torch.__version__,
                "TransformersVersion": "4.26.0",
            },
            # Tags removed - they belong on the Model Package Group, not individual packages
        }

        # Register the model
        response = sagemaker_client.create_model_package(**model_package_input)
        model_package_arn = response["ModelPackageArn"]

        print(f"✅ Model registered successfully!")
        print(f"   Model Package ARN: {model_package_arn}")
        print(f"   Model Package Group: {args.model_package_group_name}")
        print(f"   Approval Status: {args.model_approval_status}")

        # Save registration info
        registration_info = {
            "model_package_arn": model_package_arn,
            "model_package_group_name": args.model_package_group_name,
            "model_approval_status": args.model_approval_status,
            "training_job_name": training_job_name,
            "registration_timestamp": datetime.utcnow().isoformat(),
            "model_metrics": evaluation_results,
            "hyperparameters": {
                "model_name": args.model_name,
                "num_train_epochs": args.num_train_epochs,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps,
                "max_length": args.max_length,
            },
        }

        # Save to model directory
        registration_path = os.path.join(args.model_dir, "model_registry_info.json")
        with open(registration_path, "w") as f:
            json.dump(registration_info, f, indent=2)

        print(f"📝 Registration info saved to {registration_path}")

        return model_package_arn

    except Exception as e:
        print(f"❌ Error registering model: {str(e)}")
        print("⚠️ Model training completed but registration failed")
        return None


def save_evaluation_to_s3(evaluation_results, training_metadata, model_artifacts_url):
    """Save evaluation results to S3 for model registry metrics"""
    try:
        s3_client = boto3.client("s3")

        # Parse S3 URL
        s3_parts = model_artifacts_url.replace("s3://", "").split("/")
        bucket = s3_parts[0]
        key_prefix = "/".join(s3_parts[1:-1])  # Remove model.tar.gz

        # Prepare comprehensive evaluation data
        evaluation_data = {
            "version": "1.0",
            "model_name": "bert-genre-classifier",
            "evaluation_timestamp": datetime.utcnow().isoformat(),
            "metrics": {
                "accuracy": float(evaluation_results.get("eval_accuracy", 0)),
                "f1_score": float(evaluation_results.get("eval_f1", 0)),
                "precision": float(evaluation_results.get("eval_precision", 0)),
                "recall": float(evaluation_results.get("eval_recall", 0)),
                "training_loss": float(evaluation_results.get("train_loss", 0)),
                "validation_loss": float(evaluation_results.get("eval_loss", 0)),
            },
            "training_metadata": training_metadata,
            "model_quality_metrics": [
                {
                    "name": "accuracy",
                    "value": float(evaluation_results.get("eval_accuracy", 0)),
                    "standard_deviation": 0.0,
                },
                {
                    "name": "f1_score",
                    "value": float(evaluation_results.get("eval_f1", 0)),
                    "standard_deviation": 0.0,
                },
            ],
        }

        # Upload to S3
        evaluation_key = f"{key_prefix}/evaluation_results.json"
        s3_client.put_object(
            Bucket=bucket,
            Key=evaluation_key,
            Body=json.dumps(evaluation_data, indent=2),
            ContentType="application/json",
        )

        print(f"📊 Evaluation results saved to s3://{bucket}/{evaluation_key}")

    except Exception as e:
        print(f"⚠️ Warning: Could not save evaluation results to S3: {e}")


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
        report_to=None,  # Disable external reporting
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
    training_result = trainer.train()
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

    # Prepare training metadata
    training_metadata = {
        "dataset_size": len(train_df_full),
        "training_samples": len(train_texts),
        "validation_samples": len(val_texts),
        "num_labels": num_labels,
        "genres": list(unique_genres),
        "device": str(device),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "training_time_seconds": training_result.metrics.get("train_runtime", 0),
        "total_training_steps": (
            training_result.global_step
            if hasattr(training_result, "global_step")
            else 0
        ),
    }

    # Combine training and evaluation results
    all_results = {**eval_results, **training_result.metrics}

    # Determine model artifacts URL
    training_job_name = os.environ.get("SM_TRAINING_JOB_NAME", "unknown-job")
    output_s3_path = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")

    # Construct model artifacts URL (SageMaker automatically uploads to S3)
    if "s3://" in output_s3_path:
        model_artifacts_url = f"{output_s3_path.rstrip('/')}/model.tar.gz"
    else:
        # Fallback - construct based on typical SageMaker pattern
        model_artifacts_url = f"s3://sagemaker-us-east-1-{boto3.client('sts').get_caller_identity()['Account']}/sagemaker-training-job-{training_job_name}/output/model.tar.gz"

    # Register model in SageMaker Model Registry
    model_package_arn = register_model_version(
        args, model_artifacts_url, all_results, training_metadata
    )

    # Save comprehensive results
    final_results = {
        "training_completed": True,
        "model_artifacts_url": model_artifacts_url,
        "model_package_arn": model_package_arn,
        "evaluation_results": eval_results,
        "training_results": training_result.metrics,
        "training_metadata": training_metadata,
        "hyperparameters": {
            "model_name": args.model_name,
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "max_length": args.max_length,
        },
        "completion_timestamp": datetime.utcnow().isoformat(),
    }

    # Save final results
    results_path = os.path.join(args.model_dir, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"📋 Final results saved to {results_path}")
    print("🎉 Training completed successfully with model registration!")


if __name__ == "__main__":
    main()
