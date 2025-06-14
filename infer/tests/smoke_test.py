import requests
import time
import sys
import os

payload = {"data": "What is love, baby don't hurt me!"}
ip_address = os.environ.get(["PUBLIC_IP"])
url = f"http://{ip_address}:8000/predict"

for _ in range(10):
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code == 200:
            print("Smoke test passed:", r.json())
        else:
            print("App responded with status:", r.json())
    except Exception as e:
        print("Waiting for app to accept requests...", e)
        time.sleep(2)
