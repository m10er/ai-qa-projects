import requests


BASE_URL="http://127.0.0.1:5000"

import requests

BASE_URL = "http://127.0.0.1:5000"

def test_predict_spam():
    payload = {"text": "Win money now"}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == "spam"

def test_predict_not_spam():
    payload = {"text": "Let's meet for lunch"}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == "not spam"

def test_metrics_endpoint():
    response = requests.get(f"{BASE_URL}/metrics")
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["precision"] <= 1
    assert 0 <= data["recall"] <= 1
    assert 0 <= data["f1_score"] <= 1
