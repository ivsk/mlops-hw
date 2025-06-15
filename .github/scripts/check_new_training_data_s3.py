import os
import sys
from datetime import datetime, timedelta, timezone
import boto3


def find_latest_new_file(bucket_name, prefix, hours=24):
    """
    Finds the most recently uploaded file in an S3 prefix within a given time window.

    Args:
        bucket_name (str): The name of the S3 bucket.
        prefix (str): The prefix (folder) to search within.
        hours (int): The time window in hours to look for new files.

    Returns:
        str or None: The S3 URI of the latest new file, or None if no new files are found.
    """
    s3 = boto3.client("s3")
    now = datetime.now(timezone.utc)
    time_window_start = now - timedelta(hours=hours)

    print(
        f"Checking for new files in s3://{bucket_name}/{prefix} since {time_window_start} UTC..."
    )

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    latest_file = None
    latest_mod_time = None

    for page in pages:
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            mod_time = obj["LastModified"]
            if mod_time > time_window_start:
                if latest_file is None or mod_time > latest_mod_time:
                    latest_file = obj["Key"]
                    latest_mod_time = mod_time

    if latest_file:
        s3_uri = f"s3://{bucket_name}/{latest_file}"
        print(f"✅ Found new data file: {s3_uri} (Last modified: {latest_mod_time})")
        return s3_uri
    else:
        print("ℹ️ No new data files found in the last {hours} hours.")
        return None


def set_github_output(name, value):
    """Sets a GitHub Actions output."""
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"{name}={value}\n")


def main():
    bucket = os.environ["S3_BUCKET"]
    data_prefix = "raw-data/"

    latest_s3_path = find_latest_new_file(bucket, data_prefix, hours=24)

    if latest_s3_path:
        set_github_output("new_data_found", "true")
        set_github_output("data_path", latest_s3_path)
    else:
        set_github_output("new_data_found", "false")
        set_github_output("data_path", "")


if __name__ == "__main__":
    main()
