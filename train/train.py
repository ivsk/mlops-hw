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

# MLflow imports
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

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

    # MLflow specific arguments
    parser.add_argument(
        "--mlflow-experiment-name", type=str, default="bert-genre-classifier"
    )
    return parser.parse_args()


def setup_mlflow(args):
    """Setup MLflow tracking"""
    try:
        # Set MLflow tracking URI to SageMaker
        mlflow.set_tracking_uri("sagemaker://")

        # Set experiment
        mlflow.set_experiment(args.mlflow_experiment_name)

        print(f"✅ MLflow setup complete")
        print(f"   Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"   Experiment: {args.mlflow_experiment_name}")

        return True

    except Exception as e:
        print(f"⚠️ Warning: MLflow setup failed: {e}")
        print("   Continuing without MLflow tracking...")
        return False


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


def log_model_to_mlflow(model, tokenizer):
    """Log model and artifacts to MLflow"""
    try:
        components = {"model": model, "tokenizer": tokenizer}
        # Log model to MLflow
        mlflow.transformers.log_model(
            artifact_path="model", transformers_model=components, name="model"
        )
        print("✅ Model logged to MLflow")
        # Get the logged model URI
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        return model_uri
    except Exception as e:
        print(f"⚠️ Warning: Model logging to MLflow failed: {e}")
        return None


def main():
    args = parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup MLflow
    mlflow_enabled = setup_mlflow(args)

    # Start MLflow run
    run_id = None
    if mlflow_enabled:
        try:
            # Get training job name from environment
            training_job_name = os.environ.get("SM_TRAINING_JOB_NAME", "unknown-job")

            # Start MLflow run with tags
            mlflow.start_run(run_name=f"training-{training_job_name}")
            run_id = mlflow.active_run().info.run_id

            # Log Git and training context
            mlflow.set_tags(
                {
                    "sagemaker.training_job_name": training_job_name,
                    "model.framework": "pytorch",
                    "model.type": "text_classification",
                    "training.device": str(device),
                }
            )

            print(f"✅ Started MLflow run: {run_id}")

        except Exception as e:
            print(f"⚠️ Warning: Failed to start MLflow run: {e}")
            mlflow_enabled = False

    try:
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

        # Log dataset information to MLflow
        if mlflow_enabled:
            try:
                mlflow.log_params(
                    {
                        "dataset_size": len(train_df_full),
                        "num_labels": num_labels,
                        "unique_genres": list(unique_genres),
                    }
                )
                mlflow.log_param("genres_mapping", json.dumps(genre_to_id))
            except Exception as e:
                print(f"⚠️ Warning: Failed to log dataset info to MLflow: {e}")

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

        # Log hyperparameters to MLflow
        if mlflow_enabled:
            try:
                mlflow.log_params(
                    {
                        "model_name": args.model_name,
                        "max_length": args.max_length,
                        "num_train_epochs": args.num_train_epochs,
                        "per_device_train_batch_size": args.per_device_train_batch_size,
                        "per_device_eval_batch_size": args.per_device_eval_batch_size,
                        "learning_rate": args.learning_rate,
                        "weight_decay": args.weight_decay,
                        "warmup_steps": args.warmup_steps,
                        "training_samples": len(train_texts),
                        "validation_samples": len(val_texts),
                    }
                )
            except Exception as e:
                print(f"⚠️ Warning: Failed to log hyperparameters to MLflow: {e}")

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

        # Log model architecture info to MLflow
        if mlflow_enabled:
            try:
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )
                mlflow.log_params(
                    {
                        "total_parameters": total_params,
                        "trainable_parameters": trainable_params,
                        "model_architecture": "BertForSequenceClassification",
                    }
                )
            except Exception as e:
                print(f"⚠️ Warning: Failed to log model info to MLflow: {e}")

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

        try:
            for key, value in eval_results.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"final_{key}", value)

            # Log training results
            for key, value in training_result.metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"training_{key}", value)
        except Exception as e:
            print(f"⚠️ Warning: Failed to log final metrics to MLflow: {e}")

        # Save label mappings
        label_map_path = os.path.join(args.model_dir, "label_mappings.json")
        labels = {
            "genre_to_id": genre_to_id,
            "id_to_genre": id_to_genre,
        }
        with open(label_map_path, "w", encoding="UTF-8") as f:
            json.dump(labels, f)

        mlflow.log_artifact(
            label_map_path,
        )

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

        # Log model to MLflow
        model_uri = log_model_to_mlflow(model, tokenizer)

        # Save comprehensive results
        final_results = {
            "training_completed": True,
            "mlflow_run_id": run_id,
            "mlflow_model_uri": model_uri,
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

        try:
            training_summary = {
                "model_performance": {
                    "accuracy": float(eval_results.get("eval_accuracy", 0)),
                    "f1_score": float(eval_results.get("eval_f1", 0)),
                    "precision": float(eval_results.get("eval_precision", 0)),
                    "recall": float(eval_results.get("eval_recall", 0)),
                },
                "training_info": training_metadata,
                "hyperparameters": final_results["hyperparameters"],
                "git_info": {
                    "actor": args.github_actor,
                    "repository": args.github_repo,
                    "sha": args.github_sha,
                    "ref": args.github_ref,
                },
            }

            # Save training summary locally and log to MLflow
            summary_path = os.path.join(args.model_dir, "training_summary.json")
            with open(summary_path, "w") as f:
                json.dump(training_summary, f, indent=2, default=str)

            mlflow.log_artifact(summary_path, "training_info")

            print("✅ Additional artifacts logged to MLflow")

        except Exception as e:
            print(f"⚠️ Warning: Failed to log additional artifacts to MLflow: {e}")

        # Save final results
        results_path = os.path.join(args.model_dir, "training_results.json")
        with open(results_path, "w") as f:
            json.dump(final_results, f, indent=2, default=str)

        print(f"📋 Final results saved to {results_path}")
        print("🎉 Training completed successfully with MLflow tracking!")

        # Log final success status to MLflow
        try:
            mlflow.log_metric("training_success", 1)
            mlflow.set_tag("status", "completed")
        except Exception as e:
            print(f"⚠️ Warning: Failed to log final status to MLflow: {e}")

    except Exception as e:
        print(f"❌ Error during training: {str(e)}")

        try:
            mlflow.log_metric("training_success", 0)
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("error", str(e))
        except Exception as mlflow_error:
            print(f"⚠️ Warning: Failed to log error to MLflow: {mlflow_error}")

        raise

    finally:
        # End MLflow run
        if mlflow_enabled and mlflow.active_run():
            try:
                mlflow.end_run()
                print("✅ MLflow run ended successfully")
            except Exception as e:
                print(f"⚠️ Warning: Failed to end MLflow run: {e}")


if __name__ == "__main__":
    main()
