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
        boto_session = boto3.Session()
        sagemaker_session = sagemaker.Session(boto_session=boto_session)
        sagemaker_client = boto_session.client(service_name="sagemaker")
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_SERVER_ARN"))
    except Exception as e:
        logger.error(f"Error: {e}")


def set_model_alias(model_name: str, alias: str) -> None:
    setup_mlflow()
    client = mlflow.MlflowClient()

    latest_model_version = client.get_latest_versions(model_name)[0]
    client.set_registered_model_alias(model_name, alias, latest_model_version.version)

    logger.info(f"Artifacts downloaded with run ID: {latest_model_version.run_id}")
    logger.info(f"Contents of current directory: {os.listdir()}")


def main():
    parser = argparse.ArgumentParser(description="MLFlow artifact downloader")

    parser.add_argument(
        "--model_name",
        required=True,
        help="Model name associated with the artifacts to download.",
    )

    parser.add_argument("--alias", required=True, help="Alias of the model")

    args = parser.parse_args()
    set_model_alias(args.model_name, args.alias)


if __name__ == "__main__":
    main()
