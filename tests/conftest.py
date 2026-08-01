import os
import subprocess
import time
import json
import requests
import pytest
from fastapi.testclient import TestClient

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

@pytest.fixture(scope="session")
def mlflow_uri():
    return MLFLOW_URI

@pytest.fixture(scope="session")
def registered_model(mlflow_uri):
    for i in range(30):
        try:
            r = requests.get(f"{mlflow_uri}/api/2.0/mlflow/versions", timeout=5)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = mlflow_uri
    out = subprocess.check_output(["python3", "train_and_register.py"], env=env)
    data = json.loads(out.decode())
    return data

@pytest.fixture
def test_client(monkeypatch):
    from main import app
    client = TestClient(app)
    return client
