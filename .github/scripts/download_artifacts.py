import mlflow
import sagemaker
import boto3
import os
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlflow_download_artifacts")


def setup_mlflow():
    try:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_SERVER_ARN"))
    except Exception as e:
        logger.error(f"Error: {e}")


def download_artifacts(model_name: str) -> None:
    setup_mlflow()
    client = mlflow.MlflowClient()

    latest_model_version = client.get_latest_versions(model_name)[0]
    client.download_artifacts(
        run_id=latest_model_version.run_id, path="model", dst_path=os.getcwd()
    )

    logger.info(f"Artifacts downloaded with run ID: {latest_model_version.run_id}")
    logger.info(f"Contents of current directory: {os.listdir()}")


def main():
    parser = argparse.ArgumentParser(description="MLFlow artifact downloader")

    parser.add_argument(
        "--model_name",
        required=True,
        help="Model name associated with the artifacts to download.",
    )

    args = parser.parse_args()
    download_artifacts(args.model_name)


if __name__ == "__main__":
    main()
