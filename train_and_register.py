#!/usr/bin/env python3
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import json
import os

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

def register_test_model():
    mlflow.set_tracking_uri(MLFLOW_URI)
    X, y = load_iris(return_X_y=True)
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(X, y)
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(clf, "model")
        run_id = run.info.run_id
        artifact_uri = mlflow.get_artifact_uri("model")
    print(json.dumps({"run_id": run_id, "artifact_uri": artifact_uri}))

if __name__ == "__main__":
    register_test_model()
