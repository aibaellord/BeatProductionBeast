import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_generate_beat():
    response = client.post("/generate-beat/", data={"style": "hiphop", "consciousness_level": 5})
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_revenue_dashboard():
    response = client.get("/revenue/dashboard/")
    assert response.status_code == 200
    assert "total_earned" in response.json()

def test_investment_option():
    response = client.post("/investment/option/", data={"amount": 100, "user_id": "testuser"})
    assert response.status_code == 200
    assert response.json()["status"] == "success"
