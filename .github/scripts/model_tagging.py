import mlflow
import sagemaker
import boto3
import os
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlflow_tag_model")


def setup_mlflow():
    try:
        boto_session = boto3.Session()
        sagemaker_session = sagemaker.Session(boto_session=boto_session)
        sagemaker_client = boto_session.client(service_name="sagemaker")
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_SERVER_ARN"))
    except Exception as e:
        logger.error(f"Error: {e}")


def set_model_tag(model_name: str, stage: str) -> None:
    setup_mlflow()
    client = mlflow.MlflowClient()

    status_tag = {
        "status": (
            "Staging, pending validation"
            if stage == "staging"
            else "Validated, production ready."
        )
    }

    client.set_registered_model_tag(model_name, status_tag)


def main():
    parser = argparse.ArgumentParser(description="MLFlow model tagging")

    parser.add_argument(
        "--model_name",
        required=True,
        help="Model name",
    )

    parser.add_argument("--stage", required=True, help="Current stage of the workflow")

    args = parser.parse_args()
    set_model_tag(args.model_name, args.alias)


if __name__ == "__main__":
    main()
