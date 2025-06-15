#!/usr/bin/env python3
"""
Professional BERT Genre Classification Training Script for Amazon SageMaker

This script trains a BERT model for text genre classification using the Hugging Face
Transformers library with MLflow tracking and SageMaker integration.

"""

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import mlflow
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
)


class SageMakerPaths:
    """SageMaker standard paths configuration"""

    INPUT_DATA = "/opt/ml/input/data"
    OUTPUT = "/opt/ml/output"
    MODEL = "/opt/ml/model"
    HYPERPARAMETERS = "/opt/ml/input/config/hyperparameters.json"


class TrainingConfig:
    """Training configuration with validation"""

    def __init__(self, args: argparse.Namespace):
        self.model_name = args.model_name
        self.max_length = args.max_length
        self.num_epochs = args.num_train_epochs
        self.train_batch_size = args.per_device_train_batch_size
        self.eval_batch_size = args.per_device_eval_batch_size
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.warmup_steps = args.warmup_steps
        self.model_dir = args.model_dir
        self.train_dir = args.train
        self.experiment_name = args.mlflow_experiment_name

        self._validate_config()

    def _validate_config(self):
        """Validate training configuration parameters"""
        if self.learning_rate <= 0:
            raise ValueError(
                f"Learning rate must be positive, got {self.learning_rate}"
            )
        if self.num_epochs <= 0:
            raise ValueError(
                f"Number of epochs must be positive, got {self.num_epochs}"
            )
        if self.max_length <= 0:
            raise ValueError(f"Max length must be positive, got {self.max_length}")

    def to_dict(self) -> Dict:
        """Convert config to dictionary for logging"""
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "num_epochs": self.num_epochs,
            "train_batch_size": self.train_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
        }


class GenreDataset(Dataset):
    """PyTorch Dataset for genre classification"""

    def __init__(self, encodings: Dict, labels: Optional[List[int]] = None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])


class DataProcessor:
    """Handles data loading and preprocessing"""

    @staticmethod
    def load_training_data(file_path: str) -> pd.DataFrame:
        """
        Load training data from text file with format: ID ::: TITLE ::: GENRE ::: DESCRIPTION

        Args:
            file_path: Path to the training data file

        Returns:
            DataFrame with training data
        """
        logger = logging.getLogger(__name__)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Training data file not found: {file_path}")

        data = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if not lines:
                raise ValueError("Training data file is empty")

            # Skip header line
            for line_num, line in enumerate(lines[1:], start=2):
                parts = line.strip().split(" ::: ")
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
                    logger.warning(
                        f"Skipping malformed line {line_num}: {line.strip()}"
                    )

            df = pd.DataFrame(data)
            logger.info(
                f"Successfully loaded {len(df)} training samples from {file_path}"
            )
            return df

        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise

    @staticmethod
    def prepare_text_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare text data by combining title and description

        Args:
            df: Input DataFrame with TITLE and DESCRIPTION columns

        Returns:
            DataFrame with TEXT column added
        """
        df = df.copy()
        df["TEXT"] = df["TITLE"] + " [SEP] " + df["DESCRIPTION"]
        return df

    @staticmethod
    def create_label_mappings(
        genres: List[str],
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Create bidirectional label mappings

        Args:
            genres: List of unique genre labels

        Returns:
            Tuple of (genre_to_id, id_to_genre) mappings
        """
        unique_genres = sorted(list(set(genres)))
        genre_to_id = {genre: i for i, genre in enumerate(unique_genres)}
        id_to_genre = {i: genre for genre, i in genre_to_id.items()}
        return genre_to_id, id_to_genre


class MLflowManager:
    """Manages MLflow experiment tracking"""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.enabled = False
        self.run_id = None
        self.logger = logging.getLogger(__name__)

    def setup(self) -> bool:
        """Initialize MLflow tracking"""
        try:
            mlflow.set_tracking_uri("sagemaker://")
            mlflow.set_experiment(self.experiment_name)
            self.enabled = True
            self.logger.info("MLflow tracking initialized successfully")
            return True
        except Exception as e:
            self.logger.warning(
                f"MLflow setup failed: {e}. Continuing without tracking."
            )
            return False

    def start_run(self, run_name: str) -> Optional[str]:
        """Start MLflow run"""
        if not self.enabled:
            return None

        try:
            training_job_name = os.environ.get("SM_TRAINING_JOB_NAME", "unknown-job")
            mlflow.start_run(run_name=f"{run_name}-{training_job_name}")
            self.run_id = mlflow.active_run().info.run_id

            # Set standard tags
            mlflow.set_tags(
                {
                    "sagemaker.training_job_name": training_job_name,
                    "model.framework": "pytorch",
                    "model.type": "text_classification",
                    "training.device": str(
                        torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    ),
                }
            )

            self.logger.info(f"Started MLflow run: {self.run_id}")
            return self.run_id
        except Exception as e:
            self.logger.error(f"Failed to start MLflow run: {e}")
            self.enabled = False
            return None

    def log_params(self, params: Dict) -> None:
        """Log parameters to MLflow"""
        if self.enabled:
            try:
                mlflow.log_params(params)
            except Exception as e:
                self.logger.warning(f"Failed to log parameters: {e}")

    def log_metrics(self, metrics: Dict) -> None:
        """Log metrics to MLflow"""
        if self.enabled:
            try:
                mlflow.log_metrics(metrics)
            except Exception as e:
                self.logger.warning(f"Failed to log metrics: {e}")

    def log_artifact(self, artifact_path: str, artifact_name: str = None) -> None:
        """Log artifact to MLflow"""
        if self.enabled:
            try:
                mlflow.log_artifact(artifact_path, artifact_name)
            except Exception as e:
                self.logger.warning(f"Failed to log artifact: {e}")

    def log_model(self, model, tokenizer) -> Optional[str]:
        """Log model to MLflow"""
        if not self.enabled:
            return None

        try:
            components = {"model": model, "tokenizer": tokenizer}
            mlflow.transformers.log_model(
                artifact_path="model",
                transformers_model=components,
                registered_model_name="bert-genre-classifier",
            )
            model_uri = f"runs:/{self.run_id}/model"
            self.logger.info(f"Model logged to MLflow: {model_uri}")
            return model_uri
        except Exception as e:
            self.logger.error(f"Failed to log model: {e}")
            return None

    def end_run(self, success: bool = True) -> None:
        """End MLflow run"""
        if self.enabled and mlflow.active_run():
            try:
                mlflow.log_metric("training_success", 1 if success else 0)
                mlflow.set_tag("status", "completed" if success else "failed")
                mlflow.end_run()
                self.logger.info("MLflow run ended successfully")
            except Exception as e:
                self.logger.warning(f"Failed to end MLflow run: {e}")


class BERTTrainer:
    """Main training class for BERT genre classification"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        self.mlflow_manager = MLflowManager(config.experiment_name)

        # Initialize components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.genre_to_id = None
        self.id_to_genre = None

    def compute_metrics(self, pred) -> Dict[str, float]:
        """Compute evaluation metrics"""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )
        accuracy = accuracy_score(labels, preds)

        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def prepare_data(self) -> Tuple[GenreDataset, GenreDataset]:
        """Load and prepare training data"""
        self.logger.info("Loading and preparing training data")

        # Load data
        train_file = os.path.join(self.config.train_dir, "train_data.txt")
        df = DataProcessor.load_training_data(train_file)

        if df.empty:
            raise ValueError("No valid training data found")

        # Prepare text
        df = DataProcessor.prepare_text_data(df)

        # Create label mappings
        self.genre_to_id, self.id_to_genre = DataProcessor.create_label_mappings(
            df["GENRE"].tolist()
        )
        df["labels"] = df["GENRE"].map(self.genre_to_id)

        num_labels = len(self.genre_to_id)
        self.logger.info(
            f"Found {num_labels} unique genres: {list(self.genre_to_id.keys())}"
        )

        # Log dataset info to MLflow
        self.mlflow_manager.log_params(
            {
                "dataset_size": len(df),
                "num_labels": num_labels,
                "genres": list(self.genre_to_id.keys()),
            }
        )

        # Split data
        if len(df) > 1:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                df["TEXT"].tolist(),
                df["labels"].tolist(),
                test_size=0.15,
                random_state=42,
                stratify=df["labels"].tolist(),
            )
        else:
            train_texts = val_texts = df["TEXT"].tolist()
            train_labels = val_labels = df["labels"].tolist()

        self.logger.info(
            f"Training samples: {len(train_texts)}, Validation samples: {len(val_texts)}"
        )

        self.tokenizer = BertTokenizerFast.from_pretrained(self.config.model_name)

        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

        val_encodings = self.tokenizer(
            val_texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

        # Create datasets
        train_dataset = GenreDataset(train_encodings, train_labels)
        val_dataset = GenreDataset(val_encodings, val_labels)

        return train_dataset, val_dataset

    def initialize_model(self) -> None:
        """Initialize BERT model"""
        self.logger.info(f"Initializing BERT model: {self.config.model_name}")

        num_labels = len(self.genre_to_id)
        self.model = BertForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=num_labels,
            id2label=self.id_to_genre,
            label2id=self.genre_to_id,
        )

        self.model.to(self.device)

        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        self.logger.info(
            f"Model loaded with {total_params:,} total parameters ({trainable_params:,} trainable)"
        )

        self.mlflow_manager.log_params(
            {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_architecture": "BertForSequenceClassification",
            }
        )

    def setup_trainer(
        self, train_dataset: GenreDataset, val_dataset: GenreDataset
    ) -> None:
        """Setup Hugging Face Trainer"""
        training_args = TrainingArguments(
            output_dir=self.config.model_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir=os.path.join(self.config.model_dir, "logs"),
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            save_total_limit=1,
            report_to=[],  # Disable default reporting
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

    def train(self) -> Dict:
        """Execute training process"""
        self.logger.info("Starting model training")

        self.mlflow_manager.setup()
        self.mlflow_manager.start_run("bert-training")

        try:
            self.mlflow_manager.log_params(self.config.to_dict())

            train_dataset, val_dataset = self.prepare_data()

            self.initialize_model()

            self.setup_trainer(train_dataset, val_dataset)

            training_result = self.trainer.train()

            self.logger.info("Evaluating trained model")
            eval_results = self.trainer.evaluate(eval_dataset=val_dataset)

            self.mlflow_manager.log_metrics(
                {
                    f"final_{k}": v
                    for k, v in eval_results.items()
                    if isinstance(v, (int, float))
                }
            )

            self._save_artifacts(eval_results, training_result)

            # Log model to MLflow
            model_uri = self.mlflow_manager.log_model(self.model, self.tokenizer)

            self.logger.info("Training completed successfully")
            self.mlflow_manager.end_run(success=True)

            return {
                "success": True,
                "eval_results": eval_results,
                "training_results": training_result.metrics,
                "model_uri": model_uri,
            }

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.mlflow_manager.end_run(success=False)
            raise

    def _save_artifacts(self, eval_results: Dict, training_result) -> None:
        """Save training artifacts"""
        # Save label mappings
        label_mappings = {
            "genre_to_id": self.genre_to_id,
            "id_to_genre": self.id_to_genre,
        }

        label_path = os.path.join(self.config.model_dir, "label_mappings.json")
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(label_mappings, f, indent=2)

        # Save training summary
        summary = {
            "model_performance": {
                "accuracy": float(eval_results.get("eval_accuracy", 0)),
                "f1_score": float(eval_results.get("eval_f1", 0)),
                "precision": float(eval_results.get("eval_precision", 0)),
                "recall": float(eval_results.get("eval_recall", 0)),
            },
            "hyperparameters": self.config.to_dict(),
            "training_metadata": {
                "device": str(self.device),
                "completion_timestamp": datetime.utcnow().isoformat(),
                "training_time_seconds": training_result.metrics.get(
                    "train_runtime", 0
                ),
            },
        }

        summary_path = os.path.join(self.config.model_dir, "training_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self.mlflow_manager.log_artifact(label_path)
        self.mlflow_manager.log_artifact(summary_path)

        self.logger.info(f"Artifacts saved to {self.config.model_dir}")


def setup_logging() -> None:
    """Configure logging for the training script"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("/opt/ml/output/training.log"),
        ],
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="BERT Genre Classification Training")

    # Paths
    parser.add_argument(
        "--model-dir",
        type=str,
        default=SageMakerPaths.MODEL,
        help="Model output directory",
    )
    parser.add_argument(
        "--train",
        type=str,
        default=os.path.join(SageMakerPaths.INPUT_DATA, "training"),
        help="Training data directory",
    )

    # Model parameters
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="Pre-trained model name",
    )
    parser.add_argument(
        "--max-length", type=int, default=256, help="Maximum sequence length"
    )

    # Training parameters
    parser.add_argument(
        "--num-train-epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=8,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=8,
        help="Evaluation batch size per device",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--warmup-steps", type=int, default=500, help="Number of warmup steps"
    )

    # MLflow
    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default="bert-genre-classifier",
        help="MLflow experiment name",
    )

    return parser.parse_args()


def main() -> None:
    """Main training function"""
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting BERT Genre Classification Training")
    logger.info(
        f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}"
    )

    try:
        # Parse arguments and create config
        args = parse_arguments()
        config = TrainingConfig(args)

        # Initialize trainer and run training
        trainer = BERTTrainer(config)
        results = trainer.train()

        logger.info("Training pipeline completed successfully")
        logger.info(
            f"Final accuracy: {results['eval_results'].get('eval_accuracy', 0):.4f}"
        )

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
