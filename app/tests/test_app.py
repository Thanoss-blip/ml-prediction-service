import sys
import os


# flake8: noqa: E402


os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from app import app


def test_health():
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_predict_no_features():
    client = app.test_client()
    response = client.post("/predict", json={})
    assert response.status_code == 400
