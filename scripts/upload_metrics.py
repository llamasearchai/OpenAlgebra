#!/usr/bin/env python3
import csv
import requests
from datetime import datetime
import os

def upload_metrics(file_path, environment):
    with open(file_path) as f:
        reader = csv.reader(f)
        for row in reader:
            timestamp, run_id, env, score, ready = row
            payload = {
                "timestamp": timestamp,
                "run_id": run_id,
                "environment": env,
                "quality_score": float(score),
                "deployment_ready": ready == "true",
                "source": "github-actions"
            }
            send_to_metrics_service(payload, environment)

def send_to_metrics_service(payload, environment):
    api_url = f"https://metrics.{environment}.medical-ai.com/v1/telemetry"
    try:
        response = requests.post(
            api_url,
            json=payload,
            headers={"Authorization": f"Bearer {os.environ['API_TOKEN']}"}
        )
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to upload metrics: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--env", required=True)
    args = parser.parse_args()
    upload_metrics(args.file, args.env) 