import requests
import time
import sys

payload = {"data": "What is love, baby don't hurt me!"}
url = "http://localhost:8000/predict"
health_url = "http://localhost:8000"

for _ in range(1):
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code == 200:
            print("Smoke test passed:", r.json())
        else:
            print("App responded with status:", r.json())
    except Exception as e:
        print("Waiting for app to accept requests...", e)
        time.sleep(2)
