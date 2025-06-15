import requests
import time
import os
import mlflow
import boto3
import logging
from datetime import datetime, timedelta

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("comprehensive_smoke_test")

# Get configuration from environment variables
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_SERVER_ARN")
PUBLIC_IP_STAGING = os.environ.get("PUBLIC_IP_STAGING")
EC2_INSTANCE_ID = os.environ.get("EC2_INSTANCE_ID")
MODEL_NAME = os.environ.get("MODEL_NAME")

if not all([MLFLOW_TRACKING_URI, PUBLIC_IP_STAGING, EC2_INSTANCE_ID]):
    raise ValueError(
        "Missing one or more required environment variables: MLFLOW_TRACKING_SERVER_ARN, PUBLIC_IP_STAGING, EC2_INSTANCE_ID"
    )

# --- Test Parameters ---
TARGET_URL = f"http://{PUBLIC_IP_STAGING}:8000/predict"
TEST_PAYLOAD = {
    "data": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et nibh vel neque bibendum ultricies. Nunc pharetra et felis non semper."
}
NUM_REQUESTS = 20  # Number of requests to send to simulate a small load
REQUEST_TIMEOUT = 15  # Seconds


def setup_mlflow():
    """Sets the MLflow tracking URI and experiment."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.MlflowClient()
        latest_model_version = client.get_latest_versions(MODEL_NAME)[0]

    except Exception as e:
        logger.error(f"Error setting up MLflow: {e}")
        raise
    return latest_model_version


def get_ec2_metrics(instance_id: str):
    """
    Fetches CPU and Memory utilization from AWS CloudWatch for a given EC2 instance.
    """
    logger.info(f"Fetching CloudWatch metrics for instance: {instance_id}")
    try:
        cw_client = boto3.client("cloudwatch")
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=5)

        # Query for CPU Utilization (standard metric)
        cpu_response = cw_client.get_metric_data(
            MetricDataQueries=[
                {
                    "Id": "m1",
                    "MetricStat": {
                        "Metric": {
                            "Namespace": "AWS/EC2",
                            "MetricName": "CPUUtilization",
                            "Dimensions": [
                                {"Name": "InstanceId", "Value": instance_id}
                            ],
                        },
                        "Period": 60,
                        "Stat": "Average",
                    },
                    "ReturnData": True,
                }
            ],
            StartTime=start_time,
            EndTime=end_time,
        )

        # Query for Memory Utilization (custom metric from CloudWatch Agent)
        mem_response = cw_client.get_metric_data(
            MetricDataQueries=[
                {
                    "Id": "m2",
                    "MetricStat": {
                        "Metric": {
                            "Namespace": "CWAgent",
                            "MetricName": "mem_used_percent",
                            "Dimensions": [
                                {"Name": "InstanceId", "Value": instance_id}
                            ],
                        },
                        "Period": 60,
                        "Stat": "Average",
                    },
                    "ReturnData": True,
                }
            ],
            StartTime=start_time,
            EndTime=end_time,
        )

        cpu_util = (
            cpu_response["MetricDataResults"][0]["Values"][0]
            if cpu_response["MetricDataResults"][0]["Values"]
            else None
        )
        mem_util = (
            mem_response["MetricDataResults"][0]["Values"][0]
            if mem_response["MetricDataResults"][0]["Values"]
            else None
        )

        return {"avg_cpu_utilization": cpu_util, "avg_memory_utilization": mem_util}

    except Exception as e:
        logger.warning(
            f"Could not fetch CloudWatch metrics. Check permissions and if the agent is running. Error: {e}"
        )
        return {"avg_cpu_utilization": None, "avg_memory_utilization": None}


def run_smoke_test():
    """
    Runs the main smoke test logic, including performance and resource checks,
    and logs the results to MLflow.
    """
    latest_model_version = setup_mlflow()
    model_run_id = latest_model_version.run_id

    # --- Log Parameters for reproducibility ---
    mlflow.log_params({"target_url": TARGET_URL}, run_id=model_run_id)
    mlflow.log_params({"ec2_instance_id": EC2_INSTANCE_ID}, run_id=model_run_id)
    mlflow.log_params({"num_requests": NUM_REQUESTS}, run_id=model_run_id)

    successful_requests = 0
    failed_requests = 0
    latencies = []

    # --- Load and Latency Test Loop ---
    for i in range(NUM_REQUESTS):
        try:
            start_time = time.time()
            r = requests.post(TARGET_URL, json=TEST_PAYLOAD, timeout=REQUEST_TIMEOUT)
            end_time = time.time()

            latency = (end_time - start_time) * 1000  # in milliseconds
            latencies.append(latency)
            mlflow.log_metric(
                "request_latency_ms", latency, step=i, run_id=model_run_id
            )

            if r.status_code == 200:
                successful_requests += 1
                logger.info(
                    f"Request {i+1}/{NUM_REQUESTS}: SUCCESS (Status: {r.status_code}, Latency: {latency:.2f} ms)"
                )
            else:
                failed_requests += 1
                logger.warning(
                    f"Request {i+1}/{NUM_REQUESTS}: FAILED (Status: {r.status_code}, Response: {r.text})"
                )

        except requests.exceptions.RequestException as e:
            failed_requests += 1
            logger.error(f"Request {i+1}/{NUM_REQUESTS}: FAILED (Exception: {e})")

        time.sleep(0.5)  # Small delay between requests

    # --- Log Summary Metrics ---
    if latencies:
        mlflow.log_metric(
            "avg_latency_ms", sum(latencies) / len(latencies), run_id=model_run_id
        )
        mlflow.log_metric(
            "p95_latency_ms",
            sorted(latencies)[int(len(latencies) * 0.95)],
            run_id=model_run_id,
        )
        mlflow.log_metric("max_latency_ms", max(latencies), run_id=model_run_id)

    success_rate = (successful_requests / NUM_REQUESTS) * 100
    mlflow.log_metric("success_rate_percent", success_rate, run_id=model_run_id)
    logger.info(f"Test Summary: Success Rate = {success_rate:.2f}%")

    # --- Resource Utilization Test ---
    # Fetch metrics *after* the load test to see the impact
    resource_metrics = get_ec2_metrics(EC2_INSTANCE_ID)
    if resource_metrics["avg_cpu_utilization"] is not None:
        mlflow.log_metric(
            "avg_cpu_utilization",
            resource_metrics["avg_cpu_utilization"],
            run_id=model_run_id,
        )
        logger.info(
            f"Logged CPU Utilization: {resource_metrics['avg_cpu_utilization']:.2f}%"
        )

    if resource_metrics["avg_memory_utilization"] is not None:
        mlflow.log_metric(
            "avg_memory_utilization",
            resource_metrics["avg_memory_utilization"],
            run_id=model_run_id,
        )
        logger.info(
            f"Logged Memory Utilization: {resource_metrics['avg_memory_utilization']:.2f}%"
        )

    # --- Set Final Test Status Tag ---
    if success_rate > 95:
        mlflow.set_registered_model_tag(MODEL_NAME, "smoke_test_status", "PASSED")
        logger.info("Smoke test PASSED")
    else:
        mlflow.set_tag(MODEL_NAME, "smoke_test_status", "FAILED")
        logger.error("Smoke test FAILED")
        # Optionally, raise an exception to fail a CI/CD pipeline
        # raise Exception("Smoke test failed with success rate below 95%")


if __name__ == "__main__":
    run_smoke_test()
