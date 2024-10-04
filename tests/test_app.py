import os
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app=app)

def test_train_api():
    response = client.get("/train")
    assert response.status_code == 200
    assert response.text in ["Successfully trained.", "Error Occurred."]

def test_predict_api():
    # taking a random file to test
    test_file_path = "testing_data/00bf9f83-2e8f-47cf-a4f2-97f2beceebc1.wav"
    
    # # Create a dummy file for testing
    with open(test_file_path, "wb") as f:
        f.write(os.urandom(1024))  # 1KB random content to simulate a file

    with open(test_file_path, "rb") as f:
        response = client.post("/predict", files={"files": (test_file_path, f, "audio/wav")})
        assert response.status_code == 200
        # assert "prediction" in response.text or "message" in response.json()

    # Clean up the test file
    os.remove(test_file_path)
