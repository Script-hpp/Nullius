import pytest
from fastapi.testclient import TestClient
from repro_agent.api import app

client = TestClient(app)

def test_api_403_missing_or_empty_code(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "secret")
    res = client.post("/api/discover", json={"datasets": ["Cora"]})
    assert res.status_code == 403
    res = client.post("/api/discover", json={"datasets": ["Cora"], "access_code": ""})
    assert res.status_code == 403

def test_api_403_wrong_code(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "secret")
    res = client.post("/api/discover", json={"datasets": ["Cora"], "access_code": "wrong"})
    assert res.status_code == 403

def test_api_400_unknown_dataset(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "secret")
    res = client.post("/api/discover", json={"datasets": ["UnknownDataset"], "access_code": "secret"}, )
    assert res.status_code == 400

def test_api_429_running_job(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "secret")
    
    class FakeSemaphore:
        def locked(self): return True
        
    monkeypatch.setattr("repro_agent.api.job_semaphore", FakeSemaphore())
    res = client.post("/api/discover", json={"datasets": ["Cora"], "access_code": "secret"}, )
    assert res.status_code == 429
