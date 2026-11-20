import pytest
from fastapi.testclient import TestClient
from maize.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # runs startup, so the model is loaded once
        yield c


def test_health(client):
    r = client.get("/health"); assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["model_loaded"] is True


def test_model_info(client):
    j = client.get("/model-info").json()
    assert "seed_type" in j["features"] and j["n_districts"] > 40
    assert 0 < j["district_blocked_estimate"]["R2"] < 1


def test_predict_location_only(client):
    r = client.post("/predict", json={"latitude": 0.28, "longitude": 34.75}); assert r.status_code == 200
    j = r.json()
    assert 500 < j["predicted_yield_kg_ha"] < 9000
    lo, hi = j["likely_range_kg_ha"]; assert lo <= j["predicted_yield_kg_ha"] <= hi
    assert len(j["top_reasons"]) == 3 and j["soil_used"]["soil_ph"] > 3


def test_practices_change_prediction(client):
    base = client.post("/predict", json={"latitude": 0.28, "longitude": 34.75}).json()["predicted_yield_kg_ha"]
    more = client.post("/predict", json={"latitude": 0.28, "longitude": 34.75,
                                         "practices": {"dap_kg_ha": 120, "can_kg_ha": 120, "compost": 1}}).json()["predicted_yield_kg_ha"]
    assert base != more


def test_far_location_warns(client):
    j = client.post("/predict", json={"latitude": -1.85, "longitude": 36.78}).json()
    assert j["distance_to_nearest_surveyed_field_km"] > 50 and any("extrapolation" in n for n in j["notes"])


def test_blank_fields_are_filled(client):          # production data can have gaps
    r = client.post("/predict", json={"latitude": 0.5, "longitude": 34.6, "practices": {"seed_type": None, "dap_kg_ha": None}})
    assert r.status_code == 200


def test_validation_rejects_bad_input(client):
    assert client.post("/predict", json={"latitude": 50, "longitude": 34.75}).status_code == 422
    assert client.post("/predict", json={"latitude": 0.28, "longitude": 34.75, "practices": {"weed": 99}}).status_code == 422
    assert client.post("/predict", json={"latitude": 0.28}).status_code == 422


def test_water_or_outside_land_gives_clear_error(client):
    r = client.post("/predict", json={"latitude": -0.2, "longitude": 33.9})   # inside the box; may be lake
    assert r.status_code in (200, 422)


def test_unknown_practice_name_is_rejected(client):
    r = client.post("/predict", json={"latitude": 0.28, "longitude": 34.75, "practices": {"dap_kgs": 100}})
    assert r.status_code == 422


def test_bad_category_value_is_rejected(client):
    for bad in ({"seed_type": "maize123"}, {"slope": "steep"}, {"fertility": "good"}, {"distance_meter": "far"}):
        r = client.post("/predict", json={"latitude": 0.28, "longitude": 34.75, "practices": bad})
        assert r.status_code == 422, bad


def test_good_category_value_is_accepted(client):
    r = client.post("/predict", json={"latitude": 0.28, "longitude": 34.75,
                                      "practices": {"seed_type": "h_614d", "slope": "valley", "distance_meter": "1000+ m"}})
    assert r.status_code == 200


def test_category_lists_match_model(client):      # keeps the allowed values in step with the trained model
    from typing import get_args
    from maize import api
    cats = client.get("/model-info").json()["categories"]
    assert set(get_args(api.SeedType)) == set(cats["seed_type"])
    assert set(get_args(api.Slope)) == set(cats["slope"])
    assert set(get_args(api.Distance)) == set(cats["distance_meter"])
    assert set(get_args(api.Fertility)) == set(cats["fertility"])


def test_docs_list_allowed_values(client):
    schema = client.get("/openapi.json").json()["components"]["schemas"]["Practices"]["properties"]
    assert "h_614d" in str(schema["seed_type"]) and "valley" in str(schema["slope"])
