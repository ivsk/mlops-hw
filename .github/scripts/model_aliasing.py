import mlflow
import os
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlflow_model_aliasing")


def setup_mlflow():
    try:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_SERVER_ARN"))
    except Exception as e:
        logger.error(f"Error: {e}")


def set_model_alias(model_name: str, alias: str) -> None:
    setup_mlflow()
    client = mlflow.MlflowClient()

    latest_model_version = client.get_latest_versions(model_name)[0]
    client.set_registered_model_alias(model_name, alias, latest_model_version.version)

    logger.info(f"Alias {alias} set to model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="MLFlow model aliasing.")

    parser.add_argument(
        "--model_name",
        required=True,
        help="Model name.",
    )

    parser.add_argument("--alias", required=True, help="Alias of the model")

    args = parser.parse_args()
    set_model_alias(args.model_name, args.alias)


if __name__ == "__main__":
    main()
