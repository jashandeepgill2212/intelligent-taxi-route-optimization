import pytest
from unittest.mock import patch, MagicMock
import requests
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.geocoding_service import search_locations, _cache

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_geocoding_cache():
    _cache.clear()
    yield
    _cache.clear()


def test_locations_search_prioritizes_ct_university_for_ct():
    """
    Typing 'CT' must prioritize CT University, Ludhiana, Punjab over
    international results like Catalonia or Connecticut.
    """
    mock_nominatim_data = [
        {
            "display_name": "Catalunya, España",
            "lat": "41.8523",
            "lon": "1.5745",
            "address": {"country": "Spain", "country_code": "es"}
        },
        {
            "display_name": "Connecticut, United States",
            "lat": "41.6500",
            "lon": "-72.7342",
            "address": {"state": "Connecticut", "country": "United States", "country_code": "us"}
        },
        {
            "display_name": "Catania, Sicilia, Italia",
            "lat": "37.5057",
            "lon": "14.9442",
            "address": {"country": "Italy", "country_code": "it"}
        }
    ]

    with patch("backend.app.services.geocoding_service.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_nominatim_data
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        response = client.get("/locations/search?q=CT")
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        # CT University in Ludhiana, Punjab must be the top result
        top_result = data[0]
        assert "CT University" in top_result["name"]
        assert "Ludhiana" in top_result["display_name"]
        assert top_result["latitude"] == pytest.approx(30.8256, rel=1e-3)
        assert top_result["longitude"] == pytest.approx(75.6421, rel=1e-3)


def test_locations_search_ct_university_ludhiana():
    """
    Searching 'CT University Ludhiana' returns CT University in Ludhiana, Punjab.
    """
    with patch("backend.app.services.geocoding_service.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        response = client.get("/locations/search?q=CT University Ludhiana")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert "CT University" in data[0]["name"]
        assert "Ludhiana" in data[0]["display_name"]
        assert data[0]["latitude"] == pytest.approx(30.8256, rel=1e-3)
        assert data[0]["longitude"] == pytest.approx(75.6421, rel=1e-3)


def test_locations_search_ludhiana_railway_station():
    """
    Searching 'Ludhiana Railway Station' returns the railway station from Nominatim.
    """
    mock_data = [
        {
            "display_name": "Ludhiana Railway Station, Gurudev Nagar, Ludhiana, Punjab, India",
            "lat": "30.9128",
            "lon": "75.8482",
            "address": {
                "railway": "Ludhiana Railway Station",
                "city": "Ludhiana",
                "state": "Punjab",
                "country": "India",
                "country_code": "in"
            }
        }
    ]

    with patch("backend.app.services.geocoding_service.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        response = client.get("/locations/search?q=Ludhiana Railway Station")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert "Ludhiana Railway Station" in data[0]["name"]
        assert data[0]["latitude"] == pytest.approx(30.9128, rel=1e-3)
        assert data[0]["longitude"] == pytest.approx(75.8482, rel=1e-3)


def test_locations_search_international_query_remains_possible():
    """
    When explicitly searching for international places (e.g. 'Paris, France'),
    international results are returned and not stripped.
    """
    mock_data = [
        {
            "display_name": "Paris, Île-de-France, France",
            "lat": "48.8566",
            "lon": "2.3522",
            "address": {
                "city": "Paris",
                "country": "France",
                "country_code": "fr"
            }
        }
    ]

    with patch("backend.app.services.geocoding_service.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        response = client.get("/locations/search?q=Paris, France")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "Paris"
        assert data[0]["latitude"] == pytest.approx(48.8566, rel=1e-3)
        assert data[0]["longitude"] == pytest.approx(2.3522, rel=1e-3)


def test_locations_search_empty_results():
    """
    When the geocoding provider returns no results and no local landmark matches,
    endpoint returns empty list with HTTP 200 without throwing errors.
    """
    with patch("backend.app.services.geocoding_service.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        response = client.get("/locations/search?q=NonexistentPlace123456789")
        assert response.status_code == 200
        assert response.json() == []


def test_locations_search_timeout_fallback():
    """
    Provider timeout falls back gracefully to empty list or local matches without crashing.
    """
    with patch("backend.app.services.geocoding_service.requests.get", side_effect=requests.Timeout("Connection timed out")):
        response = client.get("/locations/search?q=Clock Tower Ludhiana")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


def test_locations_search_network_error_fallback():
    """
    Provider network failure falls back gracefully to empty list or local matches.
    """
    with patch("backend.app.services.geocoding_service.requests.get", side_effect=requests.RequestException("Network unreachable")):
        response = client.get("/locations/search?q=Clock Tower Ludhiana")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


def test_locations_search_query_validation():
    """
    Query too short (< 2 characters) or omitted query returns HTTP 422.
    """
    res1 = client.get("/locations/search?q=a")
    assert res1.status_code == 422
    res2 = client.get("/locations/search")
    assert res2.status_code == 422


def test_locations_search_filters_invalid_coordinates():
    """
    Invalid coordinates outside [-90, 90] or [-180, 180] are safely ignored.
    """
    mock_data = [
        {
            "display_name": "Invalid Out-of-bounds Latitude",
            "lat": "120.0",
            "lon": "75.8300",
        },
        {
            "display_name": "Valid Point, Ludhiana, Punjab, India",
            "lat": "30.9000",
            "lon": "75.8500",
            "address": {"state": "Punjab", "country": "India", "country_code": "in"}
        }
    ]
    with patch("backend.app.services.geocoding_service.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        res = client.get("/locations/search?q=Test Point")
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 1
        assert data[0]["name"] == "Valid Point"


def test_pickup_and_destination_selection_flow():
    """
    Simulates end-to-end integration:
    1. Search pickup location ('CT University Ludhiana')
    2. Search destination location ('Ludhiana Railway Station')
    3. Feed resulting coordinates directly into route calculation endpoints.
    """
    mock_nominatim_data = [
        {
            "display_name": "Ludhiana Railway Station, Gurudev Nagar, Ludhiana, Punjab, India",
            "lat": "30.9128",
            "lon": "75.8482",
            "address": {"city": "Ludhiana", "state": "Punjab", "country": "India", "country_code": "in"}
        }
    ]

    with patch("backend.app.services.geocoding_service.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_nominatim_data
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        pickup_res = client.get("/locations/search?q=CT University Ludhiana")
        assert pickup_res.status_code == 200
        pickup_loc = pickup_res.json()[0]

        dest_res = client.get("/locations/search?q=Ludhiana Railway Station")
        assert dest_res.status_code == 200
        dest_loc = dest_res.json()[0]

        # Call route calculation with the obtained coordinates
        route_req = {
            "source": {"lat": pickup_loc["latitude"], "lng": pickup_loc["longitude"]},
            "destination": {"lat": dest_loc["latitude"], "lng": dest_loc["longitude"]},
            "traffic_level": "MEDIUM",
            "optimization_objective": "balanced"
        }
        route_res = client.post("/route/dijkstra", json=route_req)
        assert route_res.status_code == 200
        route_data = route_res.json()
        assert route_data["success"] is True
        assert len(route_data["route_coordinates"]) > 0
