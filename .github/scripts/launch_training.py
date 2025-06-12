#!/usr/bin/env python3
"""
Script to launch SageMaker training job from GitHub Actions
"""

import os
import json
import time
import boto3
import sagemaker
from sagemaker.estimator import Estimator
from datetime import datetime


def get_job_name():
    """Generate unique job name"""
    timestamp = int(time.time())
    commit_hash = os.environ.get("GITHUB_SHA", "unknown")[:8]
    branch = os.environ.get("GITHUB_REF_NAME", "unknown")
    return f"bert-classifier-{branch}-{commit_hash}-{timestamp}"


def main():
    # Get environment variables
    image_uri = os.environ["IMAGE_URI"]
    sagemaker_role = os.environ["SAGEMAKER_ROLE"]
    # s3_bucket = os.environ["S3_BUCKET"]
    aws_region = os.environ["AWS_REGION"]
    instance_type = os.environ.get("INSTANCE_TYPE", "ml.m5.large")
    epochs = float(os.environ.get("EPOCHS", "0.00001"))
    use_spot = os.environ.get("USE_SPOT", "true").lower() == "true"

    # GitHub context
    github_actor = os.environ.get("GITHUB_ACTOR", "unknown")
    github_repo = os.environ.get("GITHUB_REPOSITORY", "unknown")
    github_sha = os.environ.get("GITHUB_SHA", "unknown")

    print(f"🚀 Launching SageMaker training job")
    print(f"   Image: {image_uri}")
    print(f"   Instance: {instance_type}")
    print(f"   Epochs: {epochs}")
    print(f"   Spot instances: {use_spot}")
    print(f"   Triggered by: {github_actor}")
    print(f"   Repository: {github_repo}")
    print(f"   Commit: {github_sha}")

    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session(
        boto3.session.Session(region_name="us-east-1")
    )

    # Training data paths
    input_path = "train_data.txt"
    output_path = ""

    # Hyperparameters
    hyperparameters = {
        "model-name": "bert-base-uncased",
        "max-length": 256,
        "num-train-epochs": epochs,
        "per-device-train-batch-size": 16 if "p3" in instance_type else 2,
        "per-device-eval-batch-size": 16 if "p3" in instance_type else 2,
        "learning-rate": 2e-5,
        "weight-decay": 0.01,
        "warmup-steps": 100,
    }

    # Tags for resource tracking
    tags = [
        {"Key": "Project", "Value": "bert-genre-classifier"},
        {"Key": "Environment", "Value": "cicd"},
        {"Key": "GitHubRepo", "Value": github_repo},
        {"Key": "GitHubActor", "Value": github_actor},
        {"Key": "GitHubSHA", "Value": github_sha},
        {"Key": "CreatedBy", "Value": "github-actions"},
    ]

    # Create estimator
    estimator_kwargs = {
        "image_uri": image_uri,
        "role": sagemaker_role,
        "instance_count": 1,
        "instance_type": instance_type,
        "output_path": output_path,
        "sagemaker_session": sagemaker_session,
        "hyperparameters": hyperparameters,
        "max_run": 3600 * 6,  # 6 hours max
        "tags": tags,
        "enable_sagemaker_metrics": True,
    }

    # Add spot instance configuration if requested
    if use_spot:
        estimator_kwargs.update(
            {
                "use_spot_instances": True,
                "max_wait": 3600 * 8,  # Max wait time for spot instances
            }
        )

    estimator = Estimator(**estimator_kwargs)

    # Generate job name
    job_name = get_job_name()

    print(f"🎯 Starting training job: {job_name}")

    try:
        # Start training job
        estimator.fit(
            inputs={"training": input_path},
            job_name=job_name,
            wait=False,  # Don't wait in GitHub Actions to avoid timeout
        )

        print(f"✅ Training job started successfully!")
        print(f"   Job name: {job_name}")
        print(
            f"   Monitor at: https://{aws_region}.console.aws.amazon.com/sagemaker/home?region={aws_region}#/jobs/{job_name}"
        )

        # Create results directory
        os.makedirs("training_results", exist_ok=True)

        # Save job metadata
        job_metadata = {
            "job_name": job_name,
            "image_uri": image_uri,
            "instance_type": instance_type,
            "hyperparameters": hyperparameters,
            "input_path": input_path,
            "output_path": output_path,
            "use_spot_instances": use_spot,
            "github_context": {
                "actor": github_actor,
                "repository": github_repo,
                "sha": github_sha,
                "ref": os.environ.get("GITHUB_REF", "unknown"),
            },
            "started_at": datetime.utcnow().isoformat(),
        }

        with open("training_results/job_metadata.json", "w") as f:
            json.dump(job_metadata, f, indent=2)

        # Wait a bit to get initial status
        time.sleep(30)

        # Get job status
        sagemaker_client = boto3.client("sagemaker", region_name=aws_region)
        response = sagemaker_client.describe_training_job(TrainingJobName=job_name)

        print(f"📊 Job Status: {response['TrainingJobStatus']}")

        # Save initial status
        with open("training_results/initial_status.json", "w") as f:
            # Convert datetime objects to strings for JSON serialization
            status_data = response.copy()
            for key, value in status_data.items():
                if hasattr(value, "isoformat"):
                    status_data[key] = value.isoformat()
            json.dump(status_data, f, indent=2, default=str)

        # Set outputs for subsequent GitHub Actions steps
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write(f"job_name={job_name}\n")
                f.write(f"model_data_path={estimator.output_path}\n")
                f.write(f"job_status={response['TrainingJobStatus']}\n")

        print(f"🎉 Training pipeline completed successfully!")

    except Exception as e:
        print(f"❌ Error launching training job: {str(e)}")

        # Save error information
        os.makedirs("training_results", exist_ok=True)
        error_info = {
            "error": str(e),
            "job_name": job_name,
            "timestamp": datetime.utcnow().isoformat(),
            "github_context": {
                "actor": github_actor,
                "repository": github_repo,
                "sha": github_sha,
            },
        }

        with open("training_results/error.json", "w") as f:
            json.dump(error_info, f, indent=2)

        raise


if __name__ == "__main__":
    main()
