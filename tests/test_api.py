import pytest
from unittest.mock import MagicMock

def test_predict_endpoint(monkeypatch, test_client):
    dummy = MagicMock()
    dummy.is_loaded.return_value = True
    dummy.predict.return_value = [0, 1]

    import main
    monkeypatch.setattr(main, "model_loader", dummy)
    resp = test_client.post("/predict", json={"instances": [[1,2,3,4], [5,6,7,8]]})
    assert resp.status_code == 200
    data = resp.json()
    assert "predictions" in data
    assert data["predictions"] == [0, 1]
