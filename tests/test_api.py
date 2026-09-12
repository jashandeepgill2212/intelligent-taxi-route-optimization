import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_graph_status_endpoint():
    response = client.get("/graph/status")
    assert response.status_code == 200
    data = response.json()
    assert "num_nodes" in data
    assert data["num_nodes"] > 0


def test_dijkstra_endpoint():
    req_body = {
        "source": {"lat": 30.86, "lng": 75.81},
        "destination": {"lat": 30.93, "lng": 75.88},
        "traffic_level": "MEDIUM",
        "optimization_objective": "balanced"
    }
    response = client.post("/route/dijkstra", json=req_body)
    assert response.status_code == 200
    data = response.json()
    assert data["algorithm"] == "Dijkstra"
    assert data["success"] is True
    assert len(data["route_coordinates"]) > 0


def test_compare_routes_endpoint():
    req_body = {
        "source": {"lat": 30.86, "lng": 75.81},
        "destination": {"lat": 30.93, "lng": 75.88},
        "traffic_level": "HIGH",
        "optimization_objective": "balanced"
    }
    response = client.post("/route/compare", json=req_body)
    assert response.status_code == 200
    data = response.json()
    assert "recommended_algorithm" in data
    assert "explanation" in data
    assert "Dijkstra" in data["algorithms"]
