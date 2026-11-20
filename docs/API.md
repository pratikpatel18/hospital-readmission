# API Reference

FastAPI service in `api/main.py`. Interactive docs (Swagger): **`/docs`**, OpenAPI JSON: `/openapi.json`.

Start: `uvicorn api.main:app --port 8000` (needs `python -m src.train` first; the service refuses to start without a trained model).
Base URL (local): `http://localhost:8000`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + model name + threshold |
| POST | `/predict` | Risk score, flag, band, top SHAP factors |
| GET | `/model-info` | Model, tuned params, test metrics, threshold, cost matrix |
| GET | `/explain/global` | Global SHAP importance (`?top=15`, max 50) |
| GET | `/drift` | Drift check of recent requests vs training data (`?min_rows=50`) |

Every response carries header **`X-Process-Time-ms`** (server-side time).

## POST /predict
Query: `explain` (bool, default `true`). Set `false` to skip SHAP for the fastest path.

### Request body (all fields validated; unknown fields rejected)
| Field | Type | Rules | Notes |
|---|---|---|---|
| age | int | 0-110 | |
| sex | enum | `F` `M` `O` | |
| hospital_id | enum | `PUN-KOTHRUD` `PUN-HADAPSAR` `PUN-WAKAD` `PUN-HINJEWADI` | |
| insurance | enum | `Self-pay` `Private` `Government scheme` `Corporate` | |
| admission_type | enum | `Emergency` `Elective` `Referral` | |
| discharge_disposition | enum | `Home` `Home with follow-up` `Transferred` `Left against advice` | |
| diagnosis_group | enum | `heart_failure` `copd` `diabetes` `pneumonia` `kidney_disease` `injury` `surgical` `infection_fever` `other` | free text must be mapped first (`src/cleaning.map_diagnosis`) |
| chronic_conditions | int | 0-20 | |
| los_days | float, optional | 0-365 | omit/`null` when discharge date is missing |
| prior_adm_180d | int | 0-50 | admissions in the 180 days before this one |
| prior_adm_30d | int | 0-30 | |
| opd_visits_90d | int | 0-100 | |
| n_lab_tests | int | 0-100, default 0 | |
| n_abnormal_labs | int | 0-100, default 0 | |
| hba1c, creatinine, hemoglobin, sodium, wbc | float, optional | hba1c 2-20, creatinine 0.1-20, hemoglobin 2-25, sodium 100-180, wbc 0.5-100 | omit when test not done |
| billing_total | float | 0-5,000,000 | INR, **GST counted once** |
| discharge_dow | int, optional | 0-6 (0 = Monday) | |
| admission_month | int | 1-12 | |

### Response 200
```json
{
  "readmission_probability": 0.993,
  "risk_flag": true,
  "risk_band": "High",
  "threshold": 0.51,
  "top_factors": [
    {"feature": "utilisation_score", "value": 3.91, "shap": 1.0313, "direction": "raises risk"}
  ],
  "model": "xgboost",
  "latency_ms": 23.66
}
```
- `risk_flag` = probability >= `threshold` (cost-optimal cut-off from `artifacts/threshold.json`).
- `risk_band`: **High** if >= threshold; **Medium** if >= 0.6 x threshold; else **Low**.
- `top_factors[].shap` is the contribution in log-odds; `value` is the *transformed* (scaled / one-hot) feature value.
- The probability is a **risk ranking**, not a calibrated percentage (class weights inflate scores).

### Example
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
 "age":72,"sex":"F","hospital_id":"PUN-WAKAD","insurance":"Self-pay","admission_type":"Emergency",
 "discharge_disposition":"Left against advice","diagnosis_group":"heart_failure","chronic_conditions":3,
 "los_days":8,"prior_adm_180d":2,"prior_adm_30d":1,"opd_visits_90d":4,"creatinine":1.9,
 "billing_total":98000,"discharge_dow":5,"admission_month":8}'
```
```python
import requests
r = requests.post("http://localhost:8000/predict", json=payload, timeout=5)
r.raise_for_status(); print(r.json()["risk_band"])
```

### Errors
| Status | When | Body |
|---|---|---|
| 422 | Missing field, out-of-range value, bad enum, unknown field | FastAPI validation list (`loc`, `msg`) |
| 500 | Model files missing at start-up | service will not start (message tells you to run training) |

## GET /model-info
Returns `model`, `metric`, `test` (roc_auc, pr_auc, recall, precision, cost...), `best_params`, `threshold`, `improvement_over_baseline`, `cost_matrix_inr` (FN/FP/TP/TN).

## GET /explain/global
List of `{feature, mean_abs_shap}` sorted high -> low. Plot: `artifacts/shap_summary.png`.

## GET /drift
- `{"status":"not_enough_data", ...}` until `min_rows` requests have been received (buffer is in memory, last 2,000, reset on restart).
- Otherwise `{"status":"ok"|"drift","report":[{feature,type,psi,ks_stat,ks_pvalue,drifted}]}`. Drifted features are also logged as `WARNING` in `logs/drift.log`.
- Rule: drifted if **PSI > 0.2**, or KS p < 0.01 with KS statistic > 0.1. Monitored: age, los_days, creatinine, billing_total, prior_adm_180d, opd_visits_90d, diagnosis_group, discharge_disposition.
- Offline: `python -m src.drift --new batch.csv` or `--demo`.

## Notes
- No authentication yet (see `DEPLOYMENT.md` security checklist).
- Single-patient endpoint only; for batches loop client-side or add `/predict/batch` (see `TODO.md`).
