import os
import joblib
import numpy as np
import pytest
from model_loader import ModelLoader

def test_joblib_local(tmp_path):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    clf = RandomForestClassifier(n_estimators=5, random_state=0)
    clf.fit(X, y)
    model_path = tmp_path / "m.joblib"
    joblib.dump(clf, str(model_path))

    ml = ModelLoader(model_type="joblib", model_path=str(model_path))
    ml.load_model()
    preds = ml.predict([X[0].tolist()])
    assert isinstance(preds, list)
    assert len(preds) == 1


def test_mlflow_load(registered_model, mlflow_uri):
    model_uri = registered_model["artifact_uri"]
    run_id = registered_model["run_id"]
    runs_uri = f"runs:/{run_id}/model"

    ml = ModelLoader(model_type="mlflow", model_uri=runs_uri)
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
    ml.load_model()
    preds = ml.predict([[5.1, 3.5, 1.4, 0.2]])
    assert isinstance(preds, list)
