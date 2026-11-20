import pytest
from fastapi.testclient import TestClient

VALID = dict(age=72, sex="F", hospital_id="PUN-WAKAD", insurance="Self-pay", admission_type="Emergency",
             discharge_disposition="Left against advice", diagnosis_group="heart_failure", chronic_conditions=3,
             los_days=8, prior_adm_180d=2, prior_adm_30d=1, opd_visits_90d=4, n_lab_tests=2, n_abnormal_labs=2,
             creatinine=1.9, hba1c=None, billing_total=98000, discharge_dow=5, admission_month=8)


@pytest.fixture()
def client(trained):
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_probability_flag_and_explanation(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    j = r.json()
    assert 0 <= j["readmission_probability"] <= 1
    assert j["risk_band"] in {"High", "Medium", "Low"} and isinstance(j["risk_flag"], bool)
    assert len(j["top_factors"]) == 5


def test_missing_optional_fields_are_accepted(client):
    body = {k: v for k, v in VALID.items() if k not in {"los_days", "creatinine", "discharge_dow"}}
    assert client.post("/predict", json=body).status_code == 200


@pytest.mark.parametrize("patch", [{"age": -5}, {"age": 300}, {"hospital_id": "NOPE"}, {"billing_total": -1},
                                   {"admission_month": 13}, {"unknown_field": 1}])
def test_invalid_input_is_rejected(client, patch):
    assert client.post("/predict", json={**VALID, **patch}).status_code == 422


def test_missing_required_field_is_rejected(client):
    body = dict(VALID); body.pop("age")
    assert client.post("/predict", json=body).status_code == 422


def test_latency_under_300ms(client):
    client.post("/predict", json=VALID)  # warm-up
    times = [float(client.post("/predict", json=VALID).headers["X-Process-Time-ms"]) for _ in range(10)]
    assert sorted(times)[len(times) // 2] < 300     # median


def test_drift_endpoint(client):
    assert client.get("/drift?min_rows=100000").json()["status"] == "not_enough_data"


def test_global_explanation_and_model_info(client):
    g = client.get("/explain/global?top=5").json()
    assert len(g) == 5 and {"feature", "mean_abs_shap"} <= set(g[0])
    mi = client.get("/model-info").json()
    assert 0 < mi["threshold"] < 1 and set(mi["cost_matrix_inr"]) == {"FN", "FP", "TP", "TN"}
